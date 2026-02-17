import fsspec
from datetime import datetime

def get_graf_s3(dtime):
    stime = f"{dtime:%Y%m%d}"[:-1]
    fs = fsspec.filesystem("s3", anon=True)
    
    pattern = f"s3://twc-graf-reforecast/{stime}*/"
    dirs = fs.glob(pattern)
    
    def extract_datetime(path):
        ts = path.split('/')[-1].split('_')[0]
        return datetime.strptime(ts, "%Y%m%d%H")
    
    return min(dirs, key=lambda x: abs(extract_datetime(x) - dtime))