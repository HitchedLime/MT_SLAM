import subprocess
import time

from test_config import feature_tracker_names_str
import glob



arg1 = "--tracker_config"

for i in range(21,40):
    print(f'#######RUNNING TEST  {feature_tracker_names_str[i]} {i}  ###########################')
    if mode =="SLAM":
        subprocess.run(['python3', 'main_slam.py', arg1, str(i),"--headless"])
        time.sleep(10)
        log_files = glob.glob('MT_SLAM/logs/*')

        for log_file in log_files:
            if not os.path.basename(log_file).startswith('.'):  # Check if the file is not hidden
                try:
                    os.remove(log_file)
                    print(f'Removed log file: {log_file}')
                except Exception as e:
                   print(f'Error removing file {log_file}: {e}')
    else:
        subprocess.run(['python3', 'main_vo.py', arg1, str(i)])