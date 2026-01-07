import traceback
import runpod


def handler(job):
    try:
        # твій код
        return {"ok": True}
    except Exception as e:
        tb = traceback.format_exc()
        print("HANDLER ERROR:", repr(e))
        print(tb)
        return {"ok": False, "error": str(e), "traceback": tb}


runpod.serverless.start({"handler": handler})
