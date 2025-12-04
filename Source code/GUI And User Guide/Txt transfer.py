import pandas as pd
from pathlib import Path

def batch_txt_to_xlsx(folder, pattern="*.txt", sep=None):
    """
    folder : 文件夹路径
    pattern: 要匹配的 txt 文件名，默认 *.txt
    sep    : 分隔符
             None  -> 自动按空白（空格/tab）分隔
             '\t'  -> 制表符
             ','   -> 逗号
    """
    folder = Path(folder)

    for txt_path in folder.glob(pattern):
        xlsx_path = txt_path.with_suffix(".xlsx")

        if sep is None:
            df = pd.read_csv(txt_path, sep=None, engine="python")
        else:
            df = pd.read_csv(txt_path, sep=sep)

        df.to_excel(xlsx_path, index=False)
        print(f"Saved: {xlsx_path}")


if __name__ == "__main__":
    # 这里换成你的文件夹路径，用 r"" 避免反斜杠转义
    folder = r"C:\Users\Tiezhu\OneDrive\Desktop\Disseartation\Disertation 文件verify\2023 science memeristor\Figure_s12"

    # 如果你的 txt 是空格/Tab 分隔，用 sep=None 就行
    batch_txt_to_xlsx(folder, sep=None)

    # 如果发现读出来乱掉，说明是纯 tab 分隔，可以改成：
    # batch_txt_to_xlsx(folder, sep="\t")
