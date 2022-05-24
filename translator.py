from transformers import MarianTokenizer, MarianMTModel
from opencc import OpenCC

# 初始化
tokenizer = MarianTokenizer.from_pretrained("./opus-mt-en-zh")
model = MarianMTModel.from_pretrained("./opus-mt-en-zh")
cc = OpenCC('s2t')

# 英翻簡中
text = "Hello, the dog is cute."
batch = tokenizer([text], return_tensors="pt")

generated_ids = model.generate(**batch)
result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 簡中轉繁中
print(cc.convert(result))
