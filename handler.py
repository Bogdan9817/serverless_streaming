import os
import json
from pathlib import Path
import runpod


def debug_mounts(paths_to_check=None):
    paths_to_check = paths_to_check or [
        "/",
        "/runpod-volume",
        "/runpod-volume/SyncTalk_2D",
        "/runpod-volume/SyncTalk_2D/checkpoint",
        "/runpod-volume/SyncTalk_2D/checkpoint/Judy",
        "/runpod-volume",
    ]

    info = {
        "cwd": os.getcwd(),
        "env": {
            k: os.environ.get(k)
            for k in [
                "PWD",
                "HOME",
                "WORKDIR",
                "CHECKPOINT_ROOT",
                "RUNPOD_POD_ID",
                "RUNPOD_VOLUME_PATH",
            ]
            if os.environ.get(k) is not None
        },
        "mounts": [],
        "paths": {},
        "disk": {},
    }

    # 1) Реальні mounts (джерело правди) через /proc/self/mountinfo
    mountinfo_path = "/proc/self/mountinfo"
    if os.path.exists(mountinfo_path):
        mounts = []
        with open(mountinfo_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # формат: ... <mount_point> ... - <fstype> <source> <options>
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                mount_point = parts[4]
                sep_idx = parts.index("-") if "-" in parts else -1
                fstype = (
                    parts[sep_idx + 1]
                    if sep_idx != -1 and sep_idx + 1 < len(parts)
                    else None
                )
                source = (
                    parts[sep_idx + 2]
                    if sep_idx != -1 and sep_idx + 2 < len(parts)
                    else None
                )
                mounts.append(
                    {"mount_point": mount_point, "fstype": fstype, "source": source}
                )
        # Трохи почистимо: лишимо унікальні + відсортуємо
        seen = set()
        for m in sorted(mounts, key=lambda x: x["mount_point"]):
            key = (m["mount_point"], m["fstype"], m["source"])
            if key not in seen:
                seen.add(key)
                info["mounts"].append(m)
    else:
        info["mounts_error"] = f"{mountinfo_path} not found"

    # 2) Перевірка конкретних шляхів: існує/що всередині/який FS
    for p in paths_to_check:
        pp = Path(p)
        rec = {"exists": pp.exists(), "is_dir": pp.is_dir(), "is_file": pp.is_file()}
        try:
            if pp.exists():
                st = pp.stat()
                rec["size"] = st.st_size
                rec["device"] = st.st_dev
        except Exception as e:
            rec["stat_error"] = repr(e)

        try:
            if pp.is_dir():
                # обмежимо список, щоб не заспамити лог
                rec["ls"] = sorted([x.name for x in pp.iterdir()])[:50]
        except Exception as e:
            rec["ls_error"] = repr(e)

        info["paths"][p] = rec

    # 3) Диск usage по ключових директоріях
    # (через statvfs — видно різні filesystem за різним dev / free space)
    for p in ["/", "/runpod-volume", "/data"]:
        try:
            if os.path.exists(p):
                v = os.statvfs(p)
                total = v.f_frsize * v.f_blocks
                free = v.f_frsize * v.f_bfree
                avail = v.f_frsize * v.f_bavail
                info["disk"][p] = {"total": total, "free": free, "avail": avail}
        except Exception as e:
            info["disk"][p] = {"error": repr(e)}

    print("=== DEBUG_MOUNTS ===")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print("=== /DEBUG_MOUNTS ===")


# Викликати якнайраніше:
debug_mounts()


def generator_handler(job):
    # job_input = job["input"]
    # audio_data = job_input.get("audio_data")
    # model_name = job_input.get("model_name", "Iris")
    # for frame, _ in service[model_name].generate_frames(audio_data):
    #     _, buffer = cv2.imencode(".jpg", frame)
    #     jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    #     yield {"frame": jpg_as_text}
    for i in range(10):
        yield {"number": i}


runpod.serverless.start({"handler": generator_handler, "return_aggregate_stream": True})
