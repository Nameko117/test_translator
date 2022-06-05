from transformers import MarianTokenizer, MarianMTModel
from opencc import OpenCC

# 初始化 (英翻中)
tokenizer_en2zh = MarianTokenizer.from_pretrained("./opus-mt-en-zh")
model_en2zh = MarianMTModel.from_pretrained("./opus-mt-en-zh")
cc_en2zh = OpenCC('s2t')

# 英 翻 簡中 翻繁中
text = "Hello, may I help you?"
batch = tokenizer_en2zh([text], return_tensors="pt")
generated_ids = model_en2zh.generate(**batch)
result = tokenizer_en2zh.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(cc_en2zh.convert(result))


# 初始化 (中翻英)
tokenizer_zh2en = MarianTokenizer.from_pretrained("./opus-mt-zh-en")
model_zh2en = MarianMTModel.from_pretrained("./opus-mt-zh-en")
cc_zh2en = OpenCC('t2s')

# 繁中 翻 簡中 翻英
text = "歡迎使用英文學習輔助機器人。"
text = cc_en2zh.convert(text)
batch = tokenizer_zh2en([text], return_tensors="pt")
generated_ids = model_zh2en.generate(**batch)
result = tokenizer_zh2en.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(result)
