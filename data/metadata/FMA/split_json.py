import json

def split_json(input_json, output_json, filenames):
    result = {}
    with open(input_json, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        for filename in filenames:
            if filename in data:
                result[filename] = data[filename]

    with open(output_json, 'w', encoding='utf-8') as outfile:
        json.dump(result, outfile, ensure_ascii=False, indent=2)

# 获取文件名列表
with open('data/metadata/FMA/intro_fn_list.json', 'r', encoding='utf-8') as a_file:
    a_data = json.load(a_file)
    filenames_A = a_data
with open('data/metadata/FMA/chorus_fn_list.json', 'r', encoding='utf-8') as b_file:
    b_data = json.load(b_file)
    filenames_B = b_data
with open('data/metadata/FMA/outro_fn_list.json', 'r', encoding='utf-8') as c_file:
    c_data = json.load(c_file)
    filenames_C = c_data

# 使用split_json函数划分数据
split_json('data/metadata/FMA/prompt.json', 'data/metadata/FMA/intro_annotated.json', filenames_A)
split_json('data/metadata/FMA/prompt.json', 'data/metadata/FMA/chorus_annotated.json', filenames_B)
split_json('data/metadata/FMA/prompt.json', 'data/metadata/FMA/outro_annotated.json', filenames_C)

print("Completed.")
