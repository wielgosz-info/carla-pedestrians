import os
import sys
import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
carla_root = os.environ.get('CARLA_ROOT', '')
print(f"current_dir={current_dir}")
print(f"CARLA_ROOT={carla_root}")

candidates = glob.glob(os.path.join(carla_root, 'PythonAPI', 'carla', 'agents', 'navigation', '*.py'))
print(f"candidates from CARLA_ROOT: {candidates}")

if candidates:
    agents_dir = os.path.dirname(os.path.dirname(os.path.dirname(candidates[0])))
    print(f"agents_dir: {agents_dir}")
    sys.path.insert(0, agents_dir)
    
    # 检查找到的 agents 目录
    try:
        from agents.navigation.basic_agent import BasicAgent
        print(f"BasicAgent imported from: {BasicAgent.__module__}")
    except Exception as e:
        print(f"BasicAgent import failed: {e}")
    
    # 检查 behavior_agent
    try:
        from agents.navigation.behavior_agent import BehaviorAgent as BA
        print(f"BehaviorAgent imported from: {BA.__module__}")
    except Exception as e:
        print(f"BehaviorAgent import failed: {e}")
