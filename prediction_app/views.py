from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from PIL import Image
import os
import csv
import io
import uuid

from .model_loader import predict

def index(request):
    """
    Renders the skin lesion diagnostic dashboard
    """
    context = {
        'sex_options': [
            ('Male', 'Nam'),
            ('Female', 'Nữ'),
            ('Unknown', 'Không xác định / Khác')
        ],
        'localization_options': [
            ('Scalp', 'Da đầu'),
            ('Ear', 'Tai'),
            ('Face', 'Vùng mặt'),
            ('Back', 'Vùng lưng'),
            ('Trunk', 'Thân mình'),
            ('Chest', 'Vùng ngực'),
            ('Upper extremity', 'Chi trên (Cánh tay)'),
            ('Abdomen', 'Vùng bụng'),
            ('Unknown', 'Không xác định'),
            ('Lower extremity', 'Chi dưới (Bắp chân/Đùi)'),
            ('Genital', 'Bộ phận sinh dục'),
            ('Neck', 'Vùng cổ'),
            ('Hand', 'Bàn tay'),
            ('Foot', 'Bàn chân'),
            ('Acral', 'Đầu chi (Ngón tay/chân)')
        ]
    }
    return render(request, 'prediction_app/index.html', context)

@csrf_exempt  # Exempt CSRF for the simple prediction API if needed, though AJAX will handle CSRF token
def predict_api(request):
    """
    AJAX endpoint for skin lesion prediction
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Phương thức yêu cầu không hợp lệ. Chỉ chấp nhận POST.'}, status=400)
    
    # 1. Check for image file
    if 'image' not in request.FILES:
        return JsonResponse({'success': False, 'error': 'Vui lòng tải lên một ảnh tổn thương da.'}, status=400)
        
    image_file = request.FILES['image']
    
    # 2. Extract and validate tabular parameters
    try:
        age = float(request.POST.get('age', 50))
    except ValueError:
        age = 50.0
        
    sex = request.POST.get('sex', 'Unknown').lower().strip()
    localization = request.POST.get('localization', 'Unknown').lower().strip()
    
    try:
        # 3. Open image with PIL
        pil_image = Image.open(image_file)
        
        # 4. Run PyTorch model prediction
        results = predict(pil_image, age, sex, localization)
        
        # 5. Save the image to display on the frontend results pane
        fs = FileSystemStorage()
        filename = fs.save(f"uploads/{image_file.name}", image_file)
        uploaded_file_url = fs.url(filename)
        
        # 6. Return successful response with predictions and uploaded image URL
        return JsonResponse({
            'success': True,
            'image_url': uploaded_file_url,
            'age': age,
            'sex': sex.capitalize(),
            'localization': localization.capitalize(),
            'prediction': results
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'success': False, 'error': f"Chẩn đoán thất bại: {str(e)}"}, status=500)

@csrf_exempt
def batch_predict_api(request):
    """
    AJAX endpoint for batch skin lesion prediction using a CSV metadata file and multiple images
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Phương thức yêu cầu không hợp lệ. Chỉ chấp nhận POST.'}, status=400)
        
    csv_file = request.FILES.get('csv_file')
    if not csv_file:
        return JsonResponse({'success': False, 'error': 'Không tìm thấy tệp CSV nào được tải lên.'}, status=400)
        
    # Read and parse CSV
    try:
        decoded_file = csv_file.read().decode('utf-8').splitlines()
        reader = csv.DictReader(decoded_file)
        if not reader.fieldnames:
            return JsonResponse({'success': False, 'error': 'Tệp CSV không có tiêu đề cột (headers).'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Lỗi khi xử lý tệp CSV: {str(e)}'}, status=400)
        
    # Required HAM10000 fields
    required_fields = ['image_id', 'age', 'sex', 'localization']
    field_map = {}
    for req in required_fields:
        matching_header = next((h for h in reader.fieldnames if h.strip().lower() == req), None)
        if not matching_header:
            return JsonResponse({
                'success': False,
                'error': f"Thiếu cột bắt buộc '{req}' theo cấu trúc metadata HAM10000. Các cột tìm thấy: {', '.join(reader.fieldnames)}"
            }, status=400)
        field_map[req] = matching_header
        
    # Get all uploaded images
    uploaded_images = request.FILES.getlist('images')
    image_map = {}
    for img in uploaded_images:
        name_without_ext = os.path.splitext(img.name.lower().strip())[0]
        image_map[name_without_ext] = img
        
    fs = FileSystemStorage()
    results = []
    output_rows = []
    fieldnames_out = list(reader.fieldnames) + ['predicted_class', 'confidence', 'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    total = 0
    predicted = 0
    
    for row in reader:
        total += 1
        image_id_val = row[field_map['image_id']].strip()
        clean_image_id = os.path.splitext(image_id_val.lower())[0]
        
        img_file = image_map.get(clean_image_id)
        out_row = dict(row)
        
        if img_file:
            try:
                try:
                    age_val = float(row[field_map['age']])
                except (ValueError, TypeError):
                    age_val = 50.0
                    
                sex_val = row[field_map['sex']].lower().strip()
                loc_val = row[field_map['localization']].lower().strip()
                
                # Predict
                pil_image = Image.open(img_file)
                pred_res = predict(pil_image, age_val, sex_val, loc_val)
                
                # Add output fields to the CSV row
                out_row['predicted_class'] = pred_res['top_class']
                out_row['confidence'] = f"{pred_res['top_percentage']}%"
                for class_pred in pred_res['all_predictions']:
                    out_row[class_pred['class_code']] = f"{class_pred['percentage']}%"
                    
                # Save first 15 image URLs to show as previews in front-end
                img_url = ""
                if predicted < 15:
                    saved_name = fs.save(f"uploads/batch_{uuid.uuid4().hex[:8]}_{img_file.name}", img_file)
                    img_url = fs.url(saved_name)
                    
                results.append({
                    'image_id': image_id_val,
                    'age': age_val,
                    'sex': sex_val.capitalize(),
                    'localization': loc_val.capitalize(),
                    'predicted_class': pred_res['top_class'],
                    'predicted_class_full_name': pred_res['top_full_name'],
                    'confidence': f"{pred_res['top_percentage']}%",
                    'all_predictions': pred_res['all_predictions'],
                    'image_url': img_url,
                    'success': True
                })
                predicted += 1
            except Exception as e:
                out_row['predicted_class'] = 'LỖI'
                out_row['confidence'] = str(e)
                results.append({
                    'image_id': image_id_val,
                    'success': False,
                    'error': f"Chẩn đoán lỗi: {str(e)}"
                })
        else:
            out_row['predicted_class'] = 'KÔ TÌM THẤY ẢNH'
            out_row['confidence'] = 'N/A'
            results.append({
                'image_id': image_id_val,
                'success': False,
                'error': f"Ảnh '{image_id_val}' không tìm thấy trong danh sách ảnh tải lên."
            })
        output_rows.append(out_row)
        
    # Write result CSV file
    os.makedirs(os.path.join(fs.location, 'batch_results'), exist_ok=True)
    result_filename = f"batch_results/diagnostic_results_{uuid.uuid4().hex[:12]}.csv"
    result_filepath = os.path.join(fs.location, result_filename)
    
    try:
        with open(result_filepath, 'w', newline='', encoding='utf-8') as f:
            writer_out = csv.DictWriter(f, fieldnames=fieldnames_out)
            writer_out.writeheader()
            writer_out.writerows(output_rows)
        result_url = fs.url(result_filename)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f"Không thể tạo tệp CSV kết quả: {str(e)}"}, status=500)
        
    return JsonResponse({
        'success': True,
        'total_rows': total,
        'predicted_rows': predicted,
        'results': results,
        'download_url': result_url
    })

