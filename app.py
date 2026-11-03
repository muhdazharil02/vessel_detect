import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import pandas as pd
import os
import glob
import io
import base64
import hashlib
import time
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore

# ==========================================
# --- 0. SYSTEM CONFIGURATION ---
# ==========================================
CUSTOM_MODEL_PATH = "best.pt"          # Default custom weights path
CROP_DIR = "saved_crops"               # Local full-res crop folder (for Raspberry Pi)
FIREBASE_KEY_PATH = "serviceAccountKey.json"
HEARTBEAT_INTERVAL = 30                # Seconds between online status heartbeats
ONLINE_TIMEOUT_SECONDS = 120           # Seconds before a node is considered Offline

os.makedirs(CROP_DIR, exist_ok=True)
st.set_page_config(page_title="CMS Vessel Detection System", layout="wide")


# ==========================================
# --- 1. FIREBASE INITIALIZATION & HELPERS ---
# ==========================================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

@st.cache_resource
def init_firestore():
    """Initializes Firebase using either local JSON key (Raspberry Pi) or st.secrets (Streamlit Cloud)."""
    if not firebase_admin._apps:
        if os.path.exists(FIREBASE_KEY_PATH):
            cred = credentials.Certificate(FIREBASE_KEY_PATH)
            firebase_admin.initialize_app(cred)
        elif "firebase_service_account" in st.secrets:
            key_dict = dict(st.secrets["firebase_service_account"])
            cred = credentials.Certificate(key_dict)
            firebase_admin.initialize_app(cred)
        else:
            return None
    return firestore.client()

db = init_firestore()

if db is None:
    st.error("⚠️ **Firebase Credentials Not Found!**")
    st.info(
        f"1. **On Raspberry Pi / Local PC:** Place `{FIREBASE_KEY_PATH}` in the project folder.\n"
        "2. **On Streamlit Cloud:** Add `[firebase_service_account]` to your App Secrets."
    )
    st.stop()


def seed_default_users():
    """Creates default admin and operator accounts if the users collection is empty."""
    users_ref = db.collection("users")
    if not list(users_ref.limit(1).stream()):
        now_iso = datetime.now(timezone.utc).isoformat()
        users_ref.document("admin").set({
            "username": "admin",
            "password_hash": hash_password("admin123"),
            "role": "admin",
            "site_name": "Command Centre HQ",
            "active_model": "N/A",
            "stream_source": "N/A",
            "last_seen": now_iso,
            "is_logged_in": False
        })
        users_ref.document("operator1").set({
            "username": "operator1",
            "password_hash": hash_password("vessel2026"),
            "role": "operator",
            "site_name": "Jetty Alpha (Pi-01)",
            "active_model": "yolo26n.pt",
            "stream_source": "0",
            "last_seen": "2000-01-01T00:00:00+00:00",
            "is_logged_in": False
        })

seed_default_users()


def send_heartbeat(username, active_model=None, stream_source=None, is_logged_in=True):
    """Updates operator's online timestamp and active telemetry in Firestore."""
    update_data = {
        "last_seen": datetime.now(timezone.utc).isoformat(),
        "is_logged_in": is_logged_in
    }
    if active_model:
        update_data["active_model"] = active_model
    if stream_source is not None:
        update_data["stream_source"] = str(stream_source)
    db.collection("users").document(username).update(update_data)


def check_is_online(user_doc: dict) -> bool:
    """Returns True if user is logged in and sent a heartbeat within ONLINE_TIMEOUT_SECONDS."""
    if not user_doc.get("is_logged_in", False):
        return False
    last_seen_str = user_doc.get("last_seen", "")
    if not last_seen_str:
        return False
    try:
        last_seen_dt = datetime.fromisoformat(last_seen_str)
        diff = (datetime.now(timezone.utc) - last_seen_dt).total_seconds()
        return diff <= ONLINE_TIMEOUT_SECONDS
    except Exception:
        return False


def encode_crop_to_base64(crop_rgb_array) -> str:
    """Resizes crop to a lightweight 220x220 JPEG thumbnail and returns a Base64 string (<15KB)."""
    pil_img = Image.fromarray(crop_rgb_array)
    pil_img.thumbnail((220, 220))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_crop_and_log_firestore(orig_rgb, box, source_name, model_names, model_used, track_id=None):
    """Saves full-res crop to local date folder on Pi AND pushes record + Base64 thumbnail to Firestore."""
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")

    # 1. Create local date subfolder on the Raspberry Pi
    daily_crop_dir = os.path.join(CROP_DIR, date_str)
    os.makedirs(daily_crop_dir, exist_ok=True)

    cls_id = int(box.cls[0])
    class_name = model_names[cls_id]
    conf = round(float(box.conf[0]), 2)

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    h, w, _ = orig_rgb.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop_img = orig_rgb[y1:y2, x1:x2]
    if crop_img.size == 0:
        return

    crop_filename = f"{class_name}_{track_id if track_id else 'img'}_{now.strftime('%H%M%S_%f')}.jpg"
    local_crop_path = os.path.join(daily_crop_dir, crop_filename)

    # Save full-res locally
    Image.fromarray(crop_img).save(local_crop_path)

    # Compress thumbnail for Firestore
    b64_thumb = encode_crop_to_base64(crop_img)

    # 2. Push document to Firestore
    db.collection("detections").add({
        "timestamp": timestamp_str,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_folder": date_str,
        "operator": st.session_state.username,
        "site_name": st.session_state.site_name,
        "source_name": source_name,
        "model_used": model_used,
        "track_id": track_id if track_id is not None else -1,
        "class_name": class_name,
        "confidence": conf,
        "local_crop_path": local_crop_path,
        "crop_base64": b64_thumb
    })


@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)


# ==========================================
# --- 2. SESSION STATE & LOGIN GATE ---
# ==========================================
for key, default_val in [
    ("authenticated", False),
    ("username", ""),
    ("role", ""),
    ("site_name", ""),
    ("logged_track_ids", set()),
    ("last_heartbeat", 0.0)
]:
    if key not in st.session_state:
        st.session_state[key] = default_val

def show_login():
    _, col_m, _ = st.columns([1, 1.2, 1])
    with col_m:
        st.markdown("<h2 style='text-align: center;'>🚢 CMS Vessel Detection Login</h2>", unsafe_allow_html=True)
        st.caption("Central Monitoring & Edge CCTV Detection Portal")
        with st.form("login_form"):
            u_input = st.text_input("Username").strip()
            p_input = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)
            if submitted:
                doc = db.collection("users").document(u_input).get()
                if doc.exists:
                    user_data = doc.to_dict()
                    if user_data.get("password_hash") == hash_password(p_input):
                        st.session_state.authenticated = True
                        st.session_state.username = u_input
                        st.session_state.role = user_data.get("role", "operator")
                        st.session_state.site_name = user_data.get("site_name", "Unassigned Site")
                        send_heartbeat(u_input, is_logged_in=True)
                        st.rerun()
                st.error("Invalid username or password.")
        st.info("Default accounts: `admin` / `admin123` | `operator1` / `vessel2026`")

if not st.session_state.authenticated:
    show_login()
    st.stop()


# ==========================================
# --- 3. SIDEBAR: USER PROFILE & NAVIGATION ---
# ==========================================
st.sidebar.markdown(f"### 👤 `{st.session_state.username}`")
st.sidebar.caption(f"📍 **Site:** {st.session_state.site_name}  \n🛡️ **Role:** `{st.session_state.role.upper()}`")

if st.sidebar.button("Logout", use_container_width=True):
    send_heartbeat(st.session_state.username, is_logged_in=False)
    st.session_state.authenticated = False
    st.rerun()

st.sidebar.divider()

if st.session_state.role == "admin":
    page_mode = st.sidebar.radio(
        "Navigation",
        ["🛡️ Admin Command Centre", "📹 Operator Station View"]
    )
else:
    page_mode = "📹 Operator Station View"


# ==========================================
# --- 4. PAGE A: ADMIN COMMAND CENTRE ---
# ==========================================
if page_mode == "🛡️ Admin Command Centre":
    st.title("🛡️ Command Centre — Admin Portal")
    send_heartbeat(st.session_state.username, is_logged_in=True)

    tab_status, tab_manage, tab_reports = st.tabs([
        "📡 Live Operator & Site Status",
        "👥 Manage Operators",
        "📊 Detection Reports & Gallery"
    ])

    # --- TAB 1: LIVE OPERATOR TELEMETRY ---
    with tab_status:
        col_hdr, col_btn = st.columns([4, 1])
        col_hdr.subheader("Edge Nodes & Operator Telemetry")
        if col_btn.button("🔄 Refresh Telemetry", use_container_width=True):
            st.rerun()

        users_docs = [doc.to_dict() for doc in db.collection("users").stream()]
        operators_list = [u for u in users_docs if u.get("role") == "operator"]

        if operators_list:
            online_count = sum(1 for op in operators_list if check_is_online(op))
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Registered Sites/Operators", len(operators_list))
            m2.metric("Currently Online (🟢)", online_count)
            m3.metric("Currently Offline (🔴)", len(operators_list) - online_count)

            status_rows = []
            for op in operators_list:
                is_on = check_is_online(op)
                status_rows.append({
                    "Status": "🟢 ONLINE" if is_on else "🔴 OFFLINE",
                    "Operator Username": op.get("username"),
                    "Site / Location": op.get("site_name"),
                    "Active YOLO Model": op.get("active_model", "N/A"),
                    "Video Source": op.get("stream_source", "N/A"),
                    "Last Heartbeat (UTC)": op.get("last_seen", "")[:19].replace("T", " ")
                })
            st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
            st.caption("💡 **Tip:** Use **Raspberry Pi Connect** in your browser to screen-share directly into any Online site above.")
        else:
            st.info("No operators registered yet.")

    # --- TAB 2: ADD / MANAGE OPERATORS ---
    with tab_manage:
        col_add, col_del = st.columns(2)

        with col_add:
            st.subheader("➕ Add / Update Operator")
            with st.form("add_operator_form"):
                new_user = st.text_input("Username (e.g., operator_jetty2)").strip()
                new_pass = st.text_input("Password", type="password")
                new_site = st.text_input("Assigned Site / Pi Name (e.g., Jetty Bravo - Pi 02)").strip()
                new_role = st.selectbox("Role", ["operator", "admin"])
                default_mod = st.text_input("Default YOLO Model", value="yolo26n.pt")
                save_op = st.form_submit_button("Save Account", type="primary", use_container_width=True)

                if save_op:
                    if new_user and new_pass and new_site:
                        db.collection("users").document(new_user).set({
                            "username": new_user,
                            "password_hash": hash_password(new_pass),
                            "role": new_role,
                            "site_name": new_site,
                            "active_model": default_mod,
                            "stream_source": "0",
                            "last_seen": "2000-01-01T00:00:00+00:00",
                            "is_logged_in": False
                        })
                        st.success(f"Account `{new_user}` ({new_site}) saved to Firebase!")
                        st.rerun()
                    else:
                        st.warning("Please fill in Username, Password, and Assigned Site.")

        with col_del:
            st.subheader("🗑️ Remove Operator Account")
            removable_users = [u["username"] for u in users_docs if u["username"] != st.session_state.username]
            if removable_users:
                del_target = st.selectbox("Select Account to Remove", removable_users)
                if st.button("Delete Selected Account", type="secondary"):
                    db.collection("users").document(del_target).delete()
                    st.success(f"Deleted `{del_target}`.")
                    st.rerun()
            else:
                st.caption("No other accounts available to remove.")

    # --- TAB 3: GLOBAL REPORTS & CLOUD GALLERY ---
    with tab_reports:
        st.subheader("📊 Multi-Site Detection Reports")
        det_docs = (
            db.collection("detections")
            .order_by("created_at_utc", direction=firestore.Query.DESCENDING)
            .limit(500)
            .stream()
        )
        records = []
        for d in det_docs:
            rec = d.to_dict()
            rec["doc_id"] = d.id
            records.append(rec)

        if records:
            df_all = pd.DataFrame(records)

            # Filters
            f1, f2, f3, f4 = st.columns(4)
            date_opts = ["All Dates"] + sorted(df_all["date_folder"].dropna().unique().tolist(), reverse=True)
            site_opts = ["All Sites"] + sorted(df_all["site_name"].dropna().unique().tolist())
            op_opts = ["All Operators"] + sorted(df_all["operator"].dropna().unique().tolist())
            cls_opts = ["All Classes"] + sorted(df_all["class_name"].dropna().unique().tolist())

            sel_date = f1.selectbox("Date Folder", date_opts)
            sel_site = f2.selectbox("Site / Location", site_opts)
            sel_op = f3.selectbox("Operator", op_opts)
            sel_cls = f4.selectbox("Detected Class", cls_opts)

            df_f = df_all.copy()
            if sel_date != "All Dates":
                df_f = df_f[df_f["date_folder"] == sel_date]
            if sel_site != "All Sites":
                df_f = df_f[df_f["site_name"] == sel_site]
            if sel_op != "All Operators":
                df_f = df_f[df_f["operator"] == sel_op]
            if sel_cls != "All Classes":
                df_f = df_f[df_f["class_name"] == sel_cls]

            # Summary Metrics
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Filtered Detections", len(df_f))
            k2.metric("Active Sites in Report", df_f["site_name"].nunique())
            k3.metric("Top Detected Class", df_f["class_name"].mode()[0] if not df_f.empty else "N/A")
            k4.metric("Mean Confidence", f"{df_f['confidence'].mean():.2f}" if not df_f.empty else "N/A")

            # Breakdown Charts
            if not df_f.empty:
                c_chart1, c_chart2 = st.columns(2)
                with c_chart1:
                    st.caption("**Detections by Site**")
                    st.bar_chart(df_f["site_name"].value_counts())
                with c_chart2:
                    st.caption("**Detections by Class**")
                    st.bar_chart(df_f["class_name"].value_counts())

            # Clean Table View (exclude raw Base64 column)
            display_cols = ["timestamp", "date_folder", "site_name", "operator", "model_used", "track_id", "class_name", "confidence", "source_name"]
            existing_display_cols = [c for c in display_cols if c in df_f.columns]
            st.dataframe(df_f[existing_display_cols], use_container_width=True, height=260, hide_index=True)

            # Report Export
            csv_bytes = df_f[existing_display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Filtered Report (CSV)",
                data=csv_bytes,
                file_name=f"CMS_Report_{sel_site}_{sel_date}.csv",
                mime="text/csv",
                type="primary"
            )

            # Cloud Cropped Image Gallery (Decoded directly from Firestore Base64)
            st.markdown("#### 🖼️ Cloud Cropped Detections (Synced from Edge Pis)")
            grid = st.columns(6)
            for idx, (_, row) in enumerate(df_f.head(12).iterrows()):
                with grid[idx % 6]:
                    b64_str = row.get("crop_base64")
                    if b64_str:
                        st.image(
                            f"data:image/jpeg;base64,{b64_str}",
                            caption=f"{row['class_name']} ({row['confidence']}) | {row['site_name']}",
                            use_container_width=True
                        )
        else:
            st.info("No detection records in Firebase yet.")


# ==========================================
# --- 5. PAGE B: OPERATOR STATION VIEW ---
# ==========================================
else:
    st.title(f"📹 Operator Station — {st.session_state.site_name}")

    # Sidebar Model Selection for Operator
    st.sidebar.header("🎯 Detection Model Settings")
    local_pts = glob.glob("*.pt")
    default_pts = ["yolo26n.pt", "yolo11n.pt", "yolov8n.pt"]
    model_options = list(dict.fromkeys(local_pts + default_pts)) + ["Custom Model (.pt Path)"]

    selected_opt = st.sidebar.selectbox("Select YOLO Model", model_options)
    if selected_opt == "Custom Model (.pt Path)":
        active_model_path = st.sidebar.text_input("Custom .pt Path:", value=CUSTOM_MODEL_PATH)
    else:
        active_model_path = selected_opt

    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.45, 0.05)
    vid_stride = st.sidebar.slider("Frame Stride (Higher = Faster on Raspberry Pi)", 1, 5, 2)

    # Load Model & Update Firebase Telemetry
    model = None
    if active_model_path and (active_model_path in default_pts or os.path.exists(active_model_path)):
        try:
            model = load_yolo_model(active_model_path)
            st.sidebar.success(f"Active: `{active_model_path}`")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
    else:
        st.sidebar.warning(f"Model `{active_model_path}` not found locally.")
        st.stop()

    # Optional Class Filter
    all_classes = list(model.names.values())
    chosen_names = st.sidebar.multiselect("Filter Target Classes (Optional)", all_classes)
    selected_cls_ids = [k for k, v in model.names.items() if v in chosen_names] if chosen_names else None

    # Push initial status update when Operator loads page or changes model
    send_heartbeat(st.session_state.username, active_model=active_model_path, is_logged_in=True)

    col_stream, col_logs = st.columns([1.25, 1])

    with col_stream:
        st.subheader("1. Live CCTV / Media Feed")
        source_mode = st.radio(
            "Select Input Source:",
            ["RTSP CCTV / USB Webcam", "Upload Video File", "Upload Static Image"],
            horizontal=True
        )

        # --- LIVE CCTV / WEBCAM ---
        if source_mode == "RTSP CCTV / USB Webcam":
            stream_url = st.text_input(
                "CCTV RTSP URL (e.g., rtsp://admin:pass@192.168.1.50:554/stream1) or USB Camera Index (0):",
                value="0"
            )
            run_feed = st.toggle("▶️ Start Live Detection & Cloud Logging")
            frame_placeholder = st.empty()
            telemetry_bar = st.empty()

            if run_feed:
                send_heartbeat(st.session_state.username, active_model=active_model_path, stream_source=stream_url, is_logged_in=True)
                src = int(stream_url) if stream_url.isdigit() else stream_url
                cap = cv2.VideoCapture(src)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize RTSP latency on Raspberry Pi

                if not cap.isOpened():
                    st.error("Unable to connect to CCTV/Camera source. Check Wi-Fi and RTSP URL.")
                else:
                    frame_idx = 0
                    while cap.isOpened() and run_feed:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Stream interrupted or ended.")
                            break

                        frame_idx += 1
                        if frame_idx % vid_stride != 0:
                            continue

                        # Periodic Heartbeat to Command Centre
                        now_ts = time.time()
                        if now_ts - st.session_state.last_heartbeat > HEARTBEAT_INTERVAL:
                            send_heartbeat(st.session_state.username, active_model=active_model_path, stream_source=stream_url, is_logged_in=True)
                            st.session_state.last_heartbeat = now_ts

                        results = model.track(
                            frame,
                            conf=conf_threshold,
                            classes=selected_cls_ids,
                            persist=True,
                            verbose=False
                        )
                        orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        boxes = results[0].boxes
                        if boxes is not None and boxes.id is not None:
                            for box in boxes:
                                tid = int(box.id[0])
                                if tid not in st.session_state.logged_track_ids:
                                    save_crop_and_log_firestore(
                                        orig_rgb, box, f"CCTV ({stream_url})",
                                        model.names, active_model_path, tid
                                    )
                                    st.session_state.logged_track_ids.add(tid)

                        annotated_rgb = results[0].plot()[:, :, ::-1]
                        frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
                        telemetry_bar.caption(
                            f"🟢 **Synced to Firebase** | Model: `{active_model_path}` | "
                            f"Unique Objects Logged This Session: **{len(st.session_state.logged_track_ids)}**"
                        )
                    cap.release()

        # --- VIDEO FILE ---
        elif source_mode == "Upload Video File":
            vid_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
            if vid_file is not None:
                with open("temp_op_video.mp4", "wb") as f:
                    f.write(vid_file.read())
                if st.button("Process Video & Sync to Cloud", type="primary"):
                    cap = cv2.VideoCapture("temp_op_video.mp4")
                    frame_placeholder = st.empty()
                    frame_idx = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_idx += 1
                        if frame_idx % vid_stride != 0:
                            continue
                        results = model.track(frame, conf=conf_threshold, classes=selected_cls_ids, persist=True, verbose=False)
                        orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        boxes = results[0].boxes
                        if boxes is not None and boxes.id is not None:
                            for box in boxes:
                                tid = int(box.id[0])
                                key = f"{vid_file.name}_{tid}"
                                if key not in st.session_state.logged_track_ids:
                                    save_crop_and_log_firestore(orig_rgb, box, vid_file.name, model.names, active_model_path, tid)
                                    st.session_state.logged_track_ids.add(key)
                        frame_placeholder.image(results[0].plot()[:, :, ::-1], channels="RGB", use_container_width=True)
                    cap.release()
                    st.success("Video processing complete! Records synced to Command Centre.")
                    st.rerun()

        # --- STATIC IMAGE ---
        else:
            img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            if img_file is not None:
                img = Image.open(img_file).convert("RGB")
                results = model.predict(img, conf=conf_threshold, classes=selected_cls_ids)
                st.image(results[0].plot()[:, :, ::-1], caption="Detection Preview", use_container_width=True)
                if st.button("Save Crop & Sync to Cloud", type="primary"):
                    orig_rgb = results[0].orig_img[:, :, ::-1]
                    count = 0
                    for box in results[0].boxes:
                        save_crop_and_log_firestore(orig_rgb, box, img_file.name, model.names, active_model_path, None)
                        count += 1
                    st.success(f"Synced {count} detection(s) to Firebase and local `{CROP_DIR}/YYYY-MM-DD/`!")
                    st.rerun()

    # --- OPERATOR RECENT DETECTIONS & GALLERY ---
    with col_logs:
        st.subheader("2. Site Detections & Cropped Images")
        my_docs = (
            db.collection("detections")
            .order_by("created_at_utc", direction=firestore.Query.DESCENDING)
            .limit(100)
            .stream()
        )
        my_records = [d.to_dict() for d in my_docs if d.to_dict().get("site_name") == st.session_state.site_name]

        if my_records:
            df_op = pd.DataFrame(my_records)
            dates_avail = ["All Dates"] + sorted(df_op["date_folder"].dropna().unique().tolist(), reverse=True)
            chosen_date = st.selectbox("Filter by Date Folder:", dates_avail)
            if chosen_date != "All Dates":
                df_op = df_op[df_op["date_folder"] == chosen_date]

            st.dataframe(
                df_op[["timestamp", "class_name", "confidence", "track_id", "model_used"]],
                use_container_width=True,
                height=230,
                hide_index=True
            )

            st.markdown("#### Latest Cropped Snapshots")
            g_cols = st.columns(4)
            for i, (_, r) in enumerate(df_op.head(8).iterrows()):
                with g_cols[i % 4]:
                    if r.get("crop_base64"):
                        st.image(
                            f"data:image/jpeg;base64,{r['crop_base64']}",
                            caption=f"{r['class_name']} ({r['confidence']})",
                            use_container_width=True
                        )
        else:
            st.info("No detections logged for this site yet.")