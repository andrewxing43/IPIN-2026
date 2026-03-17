import pandas as pd

def drop_columns(input_csv, output_csv, cols_to_drop):
    print(f"正在加载测试集: {input_csv} ...")
    df = pd.read_csv(input_csv, header=None)
    print(f"删除前的数据维度: {df.shape}")
    
    # 核心逻辑：精准删除指定的列
    df_cleaned = df.drop(columns=cols_to_drop)
    
    print(f"删除后的数据维度: {df_cleaned.shape}")
    
    # 保存覆盖（index=False, header=False 保持格式干净）
    df_cleaned.to_csv(output_csv, index=False, header=False)
    print(f"✅ 成功！已删除列 {cols_to_drop} 并保存至 {output_csv}")

if __name__ == "__main__":
    # 输入你的测试集，输出覆盖原文件（或者你可以写 test_cleaned.csv 防止覆盖）
    test_file = "test.csv"
    
    # 填入你刚才在 train.csv 里删掉的列号
    columns_to_delete = [112] 
    
    drop_columns(test_file, test_file, columns_to_delete)