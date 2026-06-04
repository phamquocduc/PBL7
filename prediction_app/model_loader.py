import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import threading

# Define class label mapping & details
CLASS_LABELS = {
    0: 'akiec',
    1: 'bcc',
    2: 'bkl',
    3: 'df',
    4: 'mel',
    5: 'nv',
    6: 'vasc'
}

CLASS_FULL_NAMES = {
    'akiec': 'Dày sừng ánh sáng / Bệnh Bowen (Actinic Keratosis / Bowen\'s Disease)',
    'bcc': 'Ung thư biểu mô tế bào đáy (Basal Cell Carcinoma)',
    'bkl': 'Dày sừng lành tính (Benign Keratosis)',
    'df': 'U xơ da lành tính (Dermatofibroma)',
    'mel': 'Ung thư hắc tố (Melanoma)',
    'nv': 'Nốt ruồi hắc tố lành tính (Melanocytic Nevi)',
    'vasc': 'Tổn thương mạch máu lành tính (Vascular Lesions)'
}

CLASS_DESCRIPTIONS = {
    'akiec': 'Dày sừng ánh sáng (Actinic Keratosis) là một mảng da sần sùi, đóng vảy, phát triển do tiếp xúc nhiều năm với ánh nắng mặt trời. Đây là tổn thương tiền ung thư và có thể tiến triển thành ung thư biểu mô tế bào vảy nếu không được điều trị kịp thời.',
    'bcc': 'Ung thư biểu mô tế bào đáy là loại ung thư da phổ biến nhất. Tổn thương thường xuất hiện dưới dạng nốt ngọc trai không đau, có các mạch máu nhỏ li ti nổi lên trên bề mặt hoặc một mảng phẳng màu đỏ. Bệnh phát triển chậm, hiếm khi di căn nhưng có thể xâm lấn phá hủy mô xung quanh.',
    'bkl': 'Dày sừng lành tính bao gồm dày sừng tiết bã, là một dạng tổn thương da lành tính rất phổ biến ở người lớn tuổi. Tổn thương thường có màu nâu, đen hoặc nâu nhạt ở mặt, ngực, vai hoặc lưng, bề mặt khô ráp, dạng sáp hoặc hơi gồ lên.',
    'df': 'U xơ da là một nốt sần lành tính phổ biến trên da, thường nhỏ, chắc, màu đỏ đến nâu. Bệnh thường xuất hiện nhất ở chân, vô hại và thường xảy ra sau chấn thương nhẹ như vết côn trùng cắn hoặc xước da.',
    'mel': 'Ung thư hắc tố là loại ung thư da nguy hiểm nhất, phát triển từ các tế bào sắc tố (melanocytes). Tổn thương thường bắt đầu từ nốt ruồi hoặc xuất hiện dưới dạng đốm sẫm màu bất thường, có bờ nham nhở, màu sắc loang lổ và đường kính thường lớn hơn 6mm. Phát hiện sớm và cắt bỏ kịp thời có ý nghĩa quyết định tính mạng.',
    'nv': 'Nốt ruồi hắc tố lành tính là sự tăng sinh lành tính của các tế bào sắc tố. Chúng cực kỳ phổ biến, hoàn toàn vô hại, có thể phẳng hoặc nhô cao, dạng tròn hoặc bầu dục với màu sắc đồng đều.',
    'vasc': 'Tổn thương mạch máu lành tính bao gồm u máu anh đào (cherry angioma), u sừng mạch máu (angiokeratoma) và u hạt sinh mủ (pyogenic granuloma). Đây là các bất thường lành tính của mạch máu, biểu hiện dưới dạng các nốt hoặc đốm màu đỏ, tím hoặc xanh.'
}

# Pre-trained model weights path
BEST_MODEL_PATH = '/Users/hhh/workspace/school/PBL7/ham10000_effnet_oversampled_best.pth'

class MetaBlockFusionModel(nn.Module):
    def __init__(self, tab_dim, num_classes=7, hidden_dim=256):
        super().__init__()

        # Image backbone EfficientNet-B3 (pretrained set to False since we load weights)
        self.img_net = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)

        # Tabular MLP
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, hidden_dim)
        )

        # Image projection
        self.img_proj = nn.Sequential(
            nn.Linear(1536, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Meta-Scale gate
        self.meta_scale = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Meta-Shift gate
        self.meta_shift = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, tab, return_gate=False):
        f_img_raw = self.img_net(img)     # (B, 1536)
        f_img = self.img_proj(f_img_raw)  # (B, 256)

        f_tab = self.tab_proj(tab)        # (B, 256)

        scale = self.meta_scale(f_tab)    # (B, 256)
        shift = self.meta_shift(f_tab)    # (B, 256)

        fused = f_img * scale + shift
        out = self.classifier(fused)

        if return_gate:
            return out, scale
        return out

class ModelSingleton:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_model(cls):
        with cls._lock:
            if cls._instance is None:
                print("🤖 Loading MetaBlock Skin Prediction model...")
                # tab_dim is 19
                model = MetaBlockFusionModel(tab_dim=19, num_classes=7)
                
                # Check if weights exist
                if os.path.exists(BEST_MODEL_PATH):
                    # Load on CPU (or GPU if available, but web-app is safer on CPU by default)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
                    print(f"✅ Loaded weights from {BEST_MODEL_PATH}")
                else:
                    print(f"⚠️ Warning: Checkpoint file not found at {BEST_MODEL_PATH}. Starting with random weights.")
                
                model.eval()
                cls._instance = model
            return cls._instance

# Image transformation pipeline (same as Validation in training)
transform_pipeline = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def preprocess_image(pil_image):
    """
    Converts PIL image to OpenCV format, resizes, normalizes, and returns PyTorch tensor
    """
    image_np = np.array(pil_image.convert('RGB'))
    augmented = transform_pipeline(image=image_np)
    img_tensor = augmented['image']
    return img_tensor.unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)

def get_tabular_vector(age, sex, localization):
    """
    Converts raw clinical metadata to the 19-dimensional one-hot encoded vector expected by the model.
    """
    # Age standardized: mean=51.853220169745384, scale=16.919988013393876
    age_scaled = (age - 51.853220169745384) / 16.919988013393876
    
    # Initialize all zeros (except age at index 0)
    vec = [0.0] * 19
    vec[0] = float(age_scaled)
    
    # Sex mapping (indexes 1-3)
    sex = str(sex).lower().strip()
    if sex == 'female':
        vec[1] = 1.0
    elif sex == 'male':
        vec[2] = 1.0
    else:
        vec[3] = 1.0  # unknown
        
    # Localization mapping (indexes 4-18)
    loc = str(localization).lower().strip()
    loc_map = {
        'abdomen': 4,
        'acral': 5,
        'back': 6,
        'chest': 7,
        'ear': 8,
        'face': 9,
        'foot': 10,
        'genital': 11,
        'hand': 12,
        'lower extremity': 13,
        'neck': 14,
        'scalp': 15,
        'trunk': 16,
        'unknown': 17,
        'upper extremity': 18
    }
    
    if loc in loc_map:
        vec[loc_map[loc]] = 1.0
    else:
        vec[17] = 1.0  # default to unknown
        
    return torch.tensor([vec], dtype=torch.float32)  # Add batch dimension (1, 19)

def predict(pil_image, age, sex, localization):
    """
    Predicts the skin lesion class and probabilities given the inputs.
    """
    model = ModelSingleton.get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    img_tensor = preprocess_image(pil_image).to(device)
    tab_tensor = get_tabular_vector(age, sex, localization).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor, tab_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
    # Build result details
    predictions = []
    for idx, prob in enumerate(probs):
        class_code = CLASS_LABELS[idx]
        predictions.append({
            'class_code': class_code,
            'full_name': CLASS_FULL_NAMES[class_code],
            'probability': float(prob),
            'percentage': round(float(prob) * 100, 2),
            'description': CLASS_DESCRIPTIONS[class_code]
        })
        
    # Sort predictions by probability descending
    predictions.sort(key=lambda x: x['probability'], reverse=True)
    
    return {
        'top_class': predictions[0]['class_code'],
        'top_percentage': predictions[0]['percentage'],
        'top_full_name': predictions[0]['full_name'],
        'top_description': predictions[0]['description'],
        'all_predictions': predictions
    }
