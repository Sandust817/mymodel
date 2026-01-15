import re

def calculate_accuracy_stats(file_path):
    """
    读取文件，提取所有accuracy后的浮点数，计算个数和平均值
    
    Args:
        file_path (str): 目标文件的路径
    
    Returns:
        tuple: (数值个数, 平均值)
    """
    # 初始化存储精度值的列表
    accuracy_values = []
    
    try:
        # 打开文件并读取所有内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # 使用正则表达式匹配所有 accuracy: 后面的浮点数
            # 正则表达式解释：
            # accuracy: 匹配字面量
            # (\d+\.\d+) 捕获一个或多个数字 + 小数点 + 一个或多个数字的浮点数
            pattern = r'accuracy:(\d+\.\d+)'
            matches = re.findall(pattern, content)
            
            # 将匹配到的字符串转换为浮点数
            accuracy_values = [float(match) for match in matches]
            print(accuracy_values)
            
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
        return 0, 0.0
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return 0, 0.0
    
    # 计算个数和平均值
    count = len(accuracy_values)
    if count == 0:
        average = 0.0
        print("未找到任何accuracy数值")
    else:
        average = sum(accuracy_values) / count
    
    return count, average

# 主程序
if __name__ == "__main__":
    # 请替换为你的文件路径
    FILE_PATH = "/root/sxh/mymodel2/mymodel/nohup.log"  # 这里修改成实际的文件路径
    
    # 调用函数计算统计信息
    count, average = calculate_accuracy_stats(FILE_PATH)
    
    # 输出结果
    print(f"找到的accuracy数值个数：{count}")
    # 保留8位小数输出平均值，更易读
    print(f"accuracy平均值：{average:.8f}")