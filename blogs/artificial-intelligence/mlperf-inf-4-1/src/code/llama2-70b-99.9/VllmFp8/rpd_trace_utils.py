from functools import wraps
import os

ENABLE_TRACING_RPD=bool(int(os.getenv("ENABLE_TRACING_RPD", 0)))
ENABLE_TRACING_RPD_NON_TIMED=bool(int(os.getenv("ENABLE_TRACING_RPD_NON_TIMED", 0)))

# tracing helper
def rpd_trace_range(name=""):
    def rpd_trace(fn):
        if ENABLE_TRACING_RPD:
            @wraps(fn)
            def trace_range(*args, **kwargs):
                from rpdTracerControl import rpdTracerControl
                rpd = rpdTracerControl()
                rpd.start()
                rpd.rangePush("python", f"{name+':' if name else ''}{fn.__name__}", f"")
                result = fn(*args, **kwargs)
                rpd.rangePop()
                rpd.stop()
                return result

            return trace_range
        return fn
    return rpd_trace

def rpd_trace_range_non_timed(name=""):
    def rpd_trace(fn):
        if ENABLE_TRACING_RPD_NON_TIMED:
            @wraps(fn)
            def trace_range(*args, **kwargs):
                from rpdTracerControl import rpdTracerControl
                rpd = rpdTracerControl()
                rpd.start()
                rpd.rangePush("python", f"{name+':' if name else ''}{fn.__name__}", f"")
                result = fn(*args, **kwargs)
                rpd.rangePop()
                rpd.stop()
                return result

            return trace_range
        return fn
    return rpd_trace

def rpd_trace_range_async(name=""):
    def rpd_trace(fn):
        if ENABLE_TRACING_RPD:
            @wraps(fn)
            async def trace_range(*args, **kwargs):
                from rpdTracerControl import rpdTracerControl
                rpdTracerControl.skipCreate()
                rpd = rpdTracerControl()
                rpd.start()
                rpd.rangePush("python", f"{name+':' if name else ''}{fn.__name__}", f"")
                result = await fn(*args, **kwargs)
                rpd.rangePop()
                rpd.stop()
                return result

            return trace_range
        return fn
    return rpd_trace
