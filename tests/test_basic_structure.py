"""
测试基本项目结构是否正确
"""
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_directory_structure():
    """测试目录结构是否存在"""
    required_dirs = [
        'src',
        'src/api',
        'src/config',
        'src/data',
        'src/models',
        'src/utils',
        'scripts',
        'tests',
        'docs',
        'data',
        'models',
        'logs'
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = os.path.join(project_root, dir_name)
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"[ERROR] 缺少以下目录: {', '.join(missing_dirs)}")
        return False
    else:
        print("[OK] 所有必要目录都存在")
        return True

def test_python_files():
    """测试关键Python文件是否存在"""
    required_files = [
        'src/__init__.py',
        'src/api/__init__.py',
        'src/api/main.py',
        'src/api/schemas.py',
        'src/config/__init__.py',
        'src/config/settings.py',
        'src/config/env_config.py',
        'src/data/__init__.py',
        'src/data/dataset.py',
        'src/models/__init__.py',
        'src/models/crnn_model.py',
        'src/utils/__init__.py',
        'src/utils/image_utils.py',
        'src/utils/text_utils.py',
        'src/utils/model_utils.py',
        'scripts/train.py',
        'scripts/predict.py',
        'scripts/run_api.py',
        'requirements.txt',
        'setup.py'
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(project_root, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    if missing_files:
        print(f"[ERROR] 缺少以下文件: {', '.join(missing_files)}")
        return False
    else:
        print("[OK] 所有必要文件都存在")
        return True

def test_basic_import():
    """测试基本Python导入（不导入需要外部依赖的模块）"""
    try:
        # 测试是否能导入src包
        import src
        print("[OK] 可以导入src包")
        
        # 测试各子模块的__init__.py文件是否存在
        init_files = [
            'src/api/__init__.py',
            'src/config/__init__.py',
            'src/data/__init__.py',
            'src/models/__init__.py',
            'src/utils/__init__.py'
        ]
        
        for init_file in init_files:
            file_path = os.path.join(project_root, init_file)
            if os.path.exists(file_path):
                print(f"[OK] {init_file} 存在")
            else:
                print(f"[ERROR] {init_file} 不存在")
                return False
        
        return True
    except Exception as e:
        print(f"[ERROR] 基本导入失败: {e}")
        return False

def test_all():
    """运行所有基本结构测试"""
    print("开始测试基本项目结构...")
    
    all_passed = True
    all_passed &= test_directory_structure()
    all_passed &= test_python_files()
    all_passed &= test_basic_import()
    
    if all_passed:
        print("\n[SUCCESS] 所有基本结构测试通过！")
        print("\n注意：要运行完整功能，请先安装依赖：")
        print("pip install -r requirements.txt")
    else:
        print("\n[FAILURE] 部分基本结构测试失败！")
    
    return all_passed

if __name__ == "__main__":
    test_all()