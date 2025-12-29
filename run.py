from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
from typing import List, Dict
import os
# ==================== ⭐ التعديل الأول: تحديد GPU ====================
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # استخدام GPU الثانية فقط

# ==================== دالة process_vision_info (محدثة) ====================
def process_vision_info(messages: List[dict]):
    image_inputs = []
    video_inputs = []
    for message in messages:
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item["type"] == "image":
                    image = item["image"]
                    if isinstance(image, str):
                        # فتح الصورة مع تحسين الجودة
                        image = Image.open(image).convert("RGB")
                    elif isinstance(image, Image.Image):
                        pass
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
                    image_inputs.append(image)
                elif item["type"] == "video":
                    video_inputs.append(item["video"])
    return image_inputs if image_inputs else None, video_inputs if video_inputs else None

# تحميل النموذج والمعالج
model_name = "sherif1313/Arabic-English-handwritten-OCR-v3"

# تحميل النموذج مع إعدادات محسنة
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# إعداد المعالج مع دقة أعلى لتحسين دقة OCR
processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True
)

def extract_text_from_image(image_path):
    try:
        # ✅ استخدام prompt أكثر وضوحًا وطلبا للنص الكامل
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "ارجو استخراج النص العربي كاملاً من هذه الصورة من البداية الى النهاية بدون اي اختصار ودون ذيادة او حذف. اقرأ كل المحتوى النصي الموجود في الصورة:"},
                ],
            }
        ]

        # تحضير النص والصور
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        # معالجة المدخلات مع إعدادات محسنة
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # ✅ إعدادات توليد محسنة للنصوص الطويلة
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,  # زيادة كبيرة لاستيعاب النصوص الطويلة
            min_new_tokens=50,   # الحد الأدنى لضمان عدم القص المبكر
            do_sample=False,      # للحصول على نتائج متسقة
            temperature=0.01,      # توازن بين الإبداع والاستقرار
            top_p=0.1,           # للتنوع المعتدل
            repetition_penalty=1.1,  # منع التكرار
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            num_return_sequences=1
        )

        # استخراج النص المُولَّد فقط (بدون مطالبة المستخدم)
        input_len = inputs.input_ids.shape[1]
        output_text = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True  # تحسين المسافات
        )[0]

        return output_text.strip()

    except Exception as e:
        return f"حدث خطأ أثناء معالجة الصورة: {str(e)}"

# دالة مساعدة لتحسين جودة الصور
def enhance_image_quality(image_path):
    """تحسين جودة الصورة لتحسين دقة OCR"""
    try:
        img = Image.open(image_path)
        # زيادة الدقة إذا كانت الصورة صغيرة
        if max(img.size) < 800:
            new_size = (img.size[0] * 2, img.size[1] * 2)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        return img
    except:
        return Image.open(image_path)

# مثال للاستخدام
if __name__ == "__main__":
    TEST_IMAGES_DIR = "/home/sheriff/Desktop/88/untitled folder"
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']

    image_files = [
        os.path.join(TEST_IMAGES_DIR, f)
        for f in os.listdir(TEST_IMAGES_DIR)
        if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
    ]

    if not image_files:
        print("❌ لا توجد صور في المجلد.")
        exit()

    print(f"🔍 العثور على {len(image_files)} صورة للمعالجة")
    
    for img_path in sorted(image_files):
        print(f"\n{'='*50}")
        print(f"🖼️ معالجة: {os.path.basename(img_path)}")
        print(f"{'='*50}")
        
        try:
            # ✅ استخدم الدالة المحسنة
            extracted_text = extract_text_from_image(img_path)
            
            print("📝 النص المستخرج:")
            print("-" * 40)
            print(extracted_text)
            print("-" * 40)
            
            # ✅ حساب طول النص للمقارنة
            text_length = len(extracted_text)
            print(f"📊 طول النص: {text_length} حرف")
            
        except Exception as e:
            print(f"❌ خطأ في معالجة {os.path.basename(img_path)}: {e}")
