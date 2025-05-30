import subprocess
from test_config import feature_tracker_names_str



# Arguments to pass
arg1 = "--tracker_config"

for i in range(5,40):
    print(f'#######RUNNING TEST  {feature_tracker_names_str[i]}  ###########################')
# Run the script with arguments
    subprocess.run(['python3', 'main_slam.py', arg1, str(i),"--headless"])