'''
读取训练、验证、测试集的所有语料。
'''

if __name__ == "__main__":
    files = ['train', 'valid', 'test']
    zh_path = 'corpus.zh'
    en_path = 'corpus.en'
    zh_lines = []
    en_lines = []

    for file in files:
        with open('./en-zh/' + file + '.zh', 'r') as f:
            zh_lines.extend(f.readlines())
        with open('./en-zh/' + file + '.en', 'r') as f:
            en_lines.extend(f.readlines())    

    with open(zh_path, "w") as fch:
        fch.writelines(zh_lines)

    with open(en_path, "w") as fen:
        fen.writelines(en_lines)

    # lines of Chinese: 179901
    print("lines of Chinese: ", len(zh_lines))
    # lines of English: 179901
    print("lines of English: ", len(en_lines))
    print("-------- Get Corpus ! --------")
