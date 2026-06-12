document.addEventListener('DOMContentLoaded', () => {
    // Global variable for batch files list & current batch results
    let selectedBatchFiles = [];
    let currentBatchResults = [];

    // DOM Elements - Single Mode
    const dropzone = document.getElementById('image-dropzone');
    const imageInput = document.getElementById('image-input');
    const dropzonePrompt = document.getElementById('dropzone-prompt');
    const previewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const btnRemoveImage = document.getElementById('btn-remove-image');
    
    const ageInput = document.getElementById('age-input');
    
    const form = document.getElementById('prediction-form');
    const btnSubmit = document.getElementById('btn-submit');
    const btnText = document.getElementById('btn-text');
    const btnSpinner = document.getElementById('btn-spinner');
    
    const idleState = document.getElementById('output-idle-state');
    const loadingState = document.getElementById('output-loading-state');
    const successState = document.getElementById('output-success-state');
    const batchSuccessState = document.getElementById('output-batch-success-state');
    
    const resultImg = document.getElementById('result-img');
    const topClassName = document.getElementById('top-class-name');
    const topProbabilityBadge = document.getElementById('top-probability-badge');
    const resultSexVal = document.getElementById('result-sex-val');
    const resultAgeVal = document.getElementById('result-age-val');
    const resultLocVal = document.getElementById('result-loc-val');
    const topClassDescription = document.getElementById('top-class-description');
    const distributionBarsContainer = document.getElementById('distribution-bars-container');

    // DOM Elements - Batch Mode
    const csvDropzone = document.getElementById('csv-dropzone');
    const csvInput = document.getElementById('csv-input');
    const csvPrompt = document.getElementById('csv-prompt');
    const csvPreviewContainer = document.getElementById('csv-preview-container');
    const csvFilename = document.getElementById('csv-filename');
    const btnRemoveCsv = document.getElementById('btn-remove-csv');

    const batchImagesDropzone = document.getElementById('batch-images-dropzone');
    const batchImagesInput = document.getElementById('batch-images-input');
    const batchImagesPrompt = document.getElementById('batch-images-prompt');
    const batchImagesPreviewContainer = document.getElementById('batch-images-preview-container');
    const batchImagesCount = document.getElementById('batch-images-count');
    const btnRemoveBatchImages = document.getElementById('btn-remove-batch-images');

    const batchForm = document.getElementById('batch-prediction-form');
    const btnBatchSubmit = document.getElementById('btn-batch-submit');
    const btnBatchText = document.getElementById('btn-batch-text');
    const btnBatchSpinner = document.getElementById('btn-batch-spinner');

    const batchTotalCount = document.getElementById('batch-total-count');
    const batchPredictedCount = document.getElementById('batch-predicted-count');
    const batchDownloadBtn = document.getElementById('batch-download-btn');
    const batchTableBody = document.getElementById('batch-table-body');
    const batchGridWrapper = document.getElementById('batch-grid-wrapper');
    const batchTableWrapper = document.getElementById('batch-table-wrapper');
    const viewToggleGallery = document.getElementById('view-toggle-gallery');
    const viewToggleTable = document.getElementById('view-toggle-table');

    // Translation Map for Sex and Anatomical Site values
    const translationMap = {
        'Male': 'Nam',
        'Female': 'Nữ',
        'Unknown': 'Không xác định',
        'Scalp': 'Da đầu',
        'Ear': 'Tai',
        'Face': 'Vùng mặt',
        'Back': 'Vùng lưng',
        'Trunk': 'Thân mình',
        'Chest': 'Vùng ngực',
        'Upper extremity': 'Chi trên (Cánh tay)',
        'Abdomen': 'Vùng bụng',
        'Lower extremity': 'Chi dưới (Bắp chân/Đùi)',
        'Genital': 'Bộ phận sinh dục',
        'Neck': 'Vùng cổ',
        'Hand': 'Bàn tay',
        'Foot': 'Bàn chân',
        'Acral': 'Đầu chi (Ngón tay/chân)'
    };

    function translate(word) {
        if (!word) return 'Không rõ';
        const key = word.trim();
        const foundKey = Object.keys(translationMap).find(k => k.toLowerCase() === key.toLowerCase());
        return foundKey ? translationMap[foundKey] : word;
    }

    // ==========================================
    // TAB SWITCHING LOGIC
    // ==========================================
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // Toggle active tab buttons
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Toggle active content divs
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${targetTab}-content`) {
                    content.classList.add('active');
                }
            });
            
            // Reset output view to idle when switching tabs
            showOutputState('idle');
        });
    });

    // View Toggle Switch Handlers
    if (viewToggleGallery && viewToggleTable) {
        viewToggleGallery.addEventListener('click', () => {
            viewToggleGallery.classList.add('active');
            viewToggleTable.classList.remove('active');
            if (batchGridWrapper) batchGridWrapper.style.display = 'grid';
            if (batchTableWrapper) batchTableWrapper.style.display = 'none';
        });

        viewToggleTable.addEventListener('click', () => {
            viewToggleTable.classList.add('active');
            viewToggleGallery.classList.remove('active');
            if (batchGridWrapper) batchGridWrapper.style.display = 'none';
            if (batchTableWrapper) batchTableWrapper.style.display = 'block';
        });
    }

    // ==========================================
    // SINGLE DIAGNOSIS HANDLERS
    // ==========================================
    
    // Drag and Drop Actions
    if (dropzone) {
        dropzone.addEventListener('click', () => {
            imageInput.click();
        });

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleImageSelect(e.dataTransfer.files[0]);
            }
        });
    }

    if (imageInput) {
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleImageSelect(e.target.files[0]);
            }
        });
    }

    function handleImageSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Vui lòng chọn một tệp tin hình ảnh (PNG, JPG, JPEG).');
            return;
        }
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            const imgFilenameSpan = document.getElementById('image-filename');
            if (imgFilenameSpan) {
                imgFilenameSpan.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            }
            dropzonePrompt.style.display = 'none';
            previewContainer.style.display = 'block';
        };
        reader.readAsDataURL(file);
        
        // Update input file list if drop occurred
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        imageInput.files = dataTransfer.files;
    }

    // Remove Image Action
    if (btnRemoveImage) {
        btnRemoveImage.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent dropzone trigger
            imageInput.value = '';
            imagePreview.src = '';
            previewContainer.style.display = 'none';
            dropzonePrompt.style.display = 'flex';
        });
    }

    // Form Submission (AJAX)
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            
            if (imageInput.files.length === 0) {
                alert('Vui lòng tải lên một hình ảnh trước.');
                return;
            }

            // Show loading state
            setFormLoading(true);
            showOutputState('loading');

            const formData = new FormData(form);

            fetch('/predict/', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => {
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    return response.json().then(data => {
                        if (!response.ok) {
                            throw new Error(data.error || `Lỗi hệ thống: ${response.status}`);
                        }
                        return data;
                    });
                } else {
                    return response.text().then(text => {
                        throw new Error(`Máy chủ phản hồi không đúng định dạng JSON (${response.status})`);
                    });
                }
            })
            .then(data => {
                if (data.success) {
                    renderResults(data);
                    showOutputState('success');
                } else {
                    throw new Error(data.error || 'Chẩn đoán thất bại');
                }
            })
            .catch(err => {
                console.error(err);
                alert(`Lỗi: ${err.message}`);
                showOutputState('idle');
            })
            .finally(() => {
                setFormLoading(false);
            });
        });
    }

    function setFormLoading(isLoading) {
        if (isLoading) {
            btnSubmit.disabled = true;
            btnText.style.display = 'none';
            btnSpinner.style.display = 'inline-block';
            if (ageInput) ageInput.disabled = true;
            document.getElementById('sex-select').disabled = true;
            document.getElementById('localization-select').disabled = true;
            btnRemoveImage.disabled = true;
        } else {
            btnSubmit.disabled = false;
            btnText.style.display = 'inline-flex';
            btnSpinner.style.display = 'none';
            if (ageInput) ageInput.disabled = false;
            document.getElementById('sex-select').disabled = false;
            document.getElementById('localization-select').disabled = false;
            btnRemoveImage.disabled = false;
        }
    }

    function showOutputState(state) {
        idleState.style.display = 'none';
        loadingState.style.display = 'none';
        successState.style.display = 'none';
        batchSuccessState.style.display = 'none';

        if (state === 'idle') {
            idleState.style.display = 'flex';
        } else if (state === 'loading') {
            loadingState.style.display = 'flex';
        } else if (state === 'success') {
            successState.style.display = 'block';
        } else if (state === 'batch-success') {
            batchSuccessState.style.display = 'block';
        }
    }

    function renderResults(data) {
        const pred = data.prediction;
        
        // Update summary values
        resultImg.src = data.image_url;
        topClassName.textContent = pred.top_full_name; // Show the full disease name (both English and Vietnamese translation)
        topProbabilityBadge.textContent = `${pred.top_percentage}% Độ tin cậy`;
        
        resultAgeVal.textContent = data.age;
        resultSexVal.textContent = translate(data.sex);
        resultLocVal.textContent = translate(data.localization);
        
        topClassDescription.textContent = pred.top_description;

        // Clear previous bars
        distributionBarsContainer.innerHTML = '';

        // Build distribution bars
        pred.all_predictions.forEach((item, index) => {
            const isTop = index === 0;
            const barItem = document.createElement('div');
            barItem.className = `bar-item ${isTop ? 'top-match-bar' : ''}`;

            barItem.innerHTML = `
                <div class="bar-labels">
                    <span class="bar-name">${item.full_name} <small style="color:var(--text-muted)">(${item.class_code.toUpperCase()})</small></span>
                    <span class="bar-pct ${isTop ? 'top-match-pct' : ''}">${item.percentage}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" data-width="${item.percentage}%"></div>
                </div>
            `;

            distributionBarsContainer.appendChild(barItem);
        });

        // Trigger animations after insertion
        setTimeout(() => {
            const fills = distributionBarsContainer.querySelectorAll('.bar-fill');
            fills.forEach(fill => {
                fill.style.width = fill.getAttribute('data-width');
            });
        }, 100);
    }

    // ==========================================
    // BATCH DIAGNOSTIC HANDLERS
    // ==========================================
    
    // CSV Dropzone Drag and Drop
    if (csvDropzone) {
        csvDropzone.addEventListener('click', () => {
            csvInput.click();
        });

        csvDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            csvDropzone.classList.add('dragover');
        });

        csvDropzone.addEventListener('dragleave', () => {
            csvDropzone.classList.remove('dragover');
        });

        csvDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            csvDropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleCsvSelect(e.dataTransfer.files[0]);
            }
        });
    }

    if (csvInput) {
        csvInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleCsvSelect(e.target.files[0]);
            }
        });
    }

    function handleCsvSelect(file) {
        if (!file.name.endsWith('.csv')) {
            alert('Vui lòng chọn một tệp tin CSV hợp lệ.');
            return;
        }
        csvFilename.textContent = file.name;
        csvPrompt.style.display = 'none';
        csvPreviewContainer.style.display = 'block';

        // Parse and render preview
        const reader = new FileReader();
        reader.onload = function(e) {
            const text = e.target.result;
            const lines = text.split('\n').map(line => line.trim()).filter(line => line.length > 0);
            if (lines.length > 0) {
                const headers = lines[0].split(',');
                const rows = lines.slice(1, 4).map(line => line.split(','));
                
                let previewHtml = `<div class="csv-preview-table-wrapper">
                    <table class="csv-preview-table">
                        <thead>
                            <tr>${headers.map(h => `<th>${escapeHtml(h)}</th>`).join('')}</tr>
                        </thead>
                        <tbody>
                            ${rows.map(r => `<tr>${r.map(cell => `<td>${escapeHtml(cell)}</td>`).join('')}</tr>`).join('')}
                        </tbody>
                    </table>
                    ${lines.length > 4 ? `<div class="csv-preview-more">...và ${lines.length - 4} dòng khác</div>` : ''}
                </div>`;
                
                let existingPreview = document.getElementById('csv-table-preview');
                if (existingPreview) {
                    existingPreview.innerHTML = previewHtml;
                }
            }
        };
        reader.readAsText(file);

        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        csvInput.files = dataTransfer.files;
    }

    if (btnRemoveCsv) {
        btnRemoveCsv.addEventListener('click', (e) => {
            e.stopPropagation();
            csvInput.value = '';
            csvPrompt.style.display = 'flex';
            csvPreviewContainer.style.display = 'none';
            const existingPreview = document.getElementById('csv-table-preview');
            if (existingPreview) {
                existingPreview.innerHTML = '';
            }
        });
    }

    function escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Batch Images Drag and Drop
    if (batchImagesDropzone) {
        batchImagesDropzone.addEventListener('click', () => {
            batchImagesInput.click();
        });

        batchImagesDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            batchImagesDropzone.classList.add('dragover');
        });

        batchImagesDropzone.addEventListener('dragleave', () => {
            batchImagesDropzone.classList.remove('dragover');
        });

        batchImagesDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            batchImagesDropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleBatchImagesSelect(e.dataTransfer.files);
            }
        });
    }

    if (batchImagesInput) {
        batchImagesInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleBatchImagesSelect(e.target.files);
            }
        });
    }

    function handleBatchImagesSelect(files) {
        selectedBatchFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
        if (selectedBatchFiles.length === 0) {
            alert('Không tìm thấy tệp tin hình ảnh nào trong phần chọn của bạn.');
            return;
        }
        renderBatchThumbnails();
    }

    function renderBatchThumbnails() {
        if (selectedBatchFiles.length === 0) {
            batchImagesInput.value = '';
            batchImagesPrompt.style.display = 'flex';
            batchImagesPreviewContainer.style.display = 'none';
            const thumbnailGrid = document.getElementById('batch-images-thumbnail-grid');
            if (thumbnailGrid) {
                thumbnailGrid.innerHTML = '';
            }
            return;
        }

        batchImagesCount.textContent = `Đã chọn ${selectedBatchFiles.length} ảnh`;
        batchImagesPrompt.style.display = 'none';
        batchImagesPreviewContainer.style.display = 'block';

        // Render thumbnails grid
        let thumbnailGrid = document.getElementById('batch-images-thumbnail-grid');
        if (thumbnailGrid) {
            thumbnailGrid.innerHTML = '';
        }

        const maxThumbnails = 6;
        const filesToShow = selectedBatchFiles.slice(0, maxThumbnails);
        
        filesToShow.forEach((file, idx) => {
            const thumbDiv = document.createElement('div');
            thumbDiv.className = 'thumbnail-item';
            thumbDiv.style.position = 'relative';
            
            const img = document.createElement('img');
            const reader = new FileReader();
            reader.onload = (e) => {
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
            thumbDiv.appendChild(img);
            
            // Add a floating remove button for this specific thumbnail
            const removeBtn = document.createElement('button');
            removeBtn.type = 'button';
            removeBtn.className = 'remove-btn-floating-thumbnail';
            removeBtn.title = 'Xóa ảnh này';
            removeBtn.innerHTML = '<i class="fa-solid fa-xmark"></i>';
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation(); // prevent dropzone click trigger
                removeImageFromBatch(idx);
            });
            thumbDiv.appendChild(removeBtn);
            
            if (idx === maxThumbnails - 1 && selectedBatchFiles.length > maxThumbnails) {
                const overlay = document.createElement('div');
                overlay.className = 'thumbnail-overlay';
                overlay.textContent = `+${selectedBatchFiles.length - maxThumbnails}`;
                thumbDiv.appendChild(overlay);
            }
            
            thumbnailGrid.appendChild(thumbDiv);
        });

        // Sync files to batchImagesInput.files
        const dataTransfer = new DataTransfer();
        selectedBatchFiles.forEach(f => dataTransfer.items.add(f));
        batchImagesInput.files = dataTransfer.files;
    }

    function removeImageFromBatch(index) {
        selectedBatchFiles.splice(index, 1);
        renderBatchThumbnails();
    }

    if (btnRemoveBatchImages) {
        btnRemoveBatchImages.addEventListener('click', (e) => {
            e.stopPropagation();
            selectedBatchFiles = [];
            renderBatchThumbnails();
        });
    }

    // Batch Form submission
    if (batchForm) {
        batchForm.addEventListener('submit', (e) => {
            e.preventDefault();

            if (csvInput.files.length === 0) {
                alert('Vui lòng chọn một tệp CSV chứa thông tin lâm sàng.');
                return;
            }
            if (batchImagesInput.files.length === 0) {
                alert('Vui lòng chọn các ảnh tổn thương da.');
                return;
            }

            setBatchFormLoading(true);
            showOutputState('loading');

            const formData = new FormData(batchForm);

            fetch('/predict-batch/', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => {
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    return response.json().then(data => {
                        if (!response.ok) {
                            throw new Error(data.error || `Lỗi hệ thống: ${response.status}`);
                        }
                        return data;
                    });
                } else {
                    return response.text().then(text => {
                        throw new Error(`Máy chủ phản hồi không đúng định dạng JSON (${response.status})`);
                    });
                }
            })
            .then(data => {
                if (data.success) {
                    renderBatchResults(data);
                    showOutputState('batch-success');
                } else {
                    throw new Error(data.error || 'Chẩn đoán hàng loạt thất bại');
                }
            })
            .catch(err => {
                console.error(err);
                alert(`Lỗi: ${err.message}`);
                showOutputState('idle');
            })
            .finally(() => {
                setBatchFormLoading(false);
            });
        });
    }

    function setBatchFormLoading(isLoading) {
        if (isLoading) {
            btnBatchSubmit.disabled = true;
            btnBatchText.style.display = 'none';
            btnBatchSpinner.style.display = 'inline-block';
            if (btnRemoveCsv) btnRemoveCsv.disabled = true;
            if (btnRemoveBatchImages) btnRemoveBatchImages.disabled = true;
        } else {
            btnBatchSubmit.disabled = false;
            btnBatchText.style.display = 'inline-flex';
            btnBatchSpinner.style.display = 'none';
            if (btnRemoveCsv) btnRemoveCsv.disabled = false;
            if (btnRemoveBatchImages) btnRemoveBatchImages.disabled = false;
        }
    }

    function renderBatchResults(data) {
        currentBatchResults = data.results || [];
        batchTotalCount.textContent = data.total_rows;
        batchPredictedCount.textContent = data.predicted_rows;
        batchDownloadBtn.href = data.download_url;

        // Reset view toggle to Gallery default
        if (viewToggleGallery && viewToggleTable) {
            viewToggleGallery.classList.add('active');
            viewToggleTable.classList.remove('active');
            if (batchGridWrapper) batchGridWrapper.style.display = 'grid';
            if (batchTableWrapper) batchTableWrapper.style.display = 'none';
        }

        // Clear previous results
        if (batchTableBody) batchTableBody.innerHTML = '';
        if (batchGridWrapper) batchGridWrapper.innerHTML = '';

        const classCodeNames = {
            'akiec': 'Dày sừng ánh sáng / Bệnh Bowen (Actinic Keratosis / Bowen\'s Disease)',
            'bcc': 'Ung thư biểu mô tế bào đáy (Basal Cell Carcinoma)',
            'bkl': 'Dày sừng lành tính (Benign Keratosis)',
            'df': 'U xơ da lành tính (Dermatofibroma)',
            'mel': 'Ung thư hắc tố (Melanoma)',
            'nv': 'Nốt ruồi hắc tố lành tính (Melanocytic Nevi)',
            'vasc': 'Tổn thương mạch máu lành tính (Vascular Lesions)'
        };

        function translateClassCode(code) {
            if (!code) return 'Không rõ';
            return classCodeNames[code.toLowerCase()] || code;
        }

        data.results.forEach((res, index) => {
            // 1. Render Table row
            if (batchTableBody) {
                const tr = document.createElement('tr');
                tr.setAttribute('data-index', index);
                
                let predictionBadge = '';
                if (res.success) {
                    const cls = res.predicted_class.toLowerCase();
                    let badgeClass = 'benign';
                    if (cls === 'mel' || cls === 'bcc') {
                        badgeClass = 'dangerous';
                    } else if (cls === 'akiec') {
                        badgeClass = 'suspicious';
                    }
                    const fullName = res.predicted_class_full_name || translateClassCode(cls);
                    predictionBadge = `<div style="display:flex;flex-direction:column;gap:0.25rem;">
                        <span class="prediction-cell-badge ${badgeClass}" style="width:fit-content;">${res.predicted_class.toUpperCase()}</span>
                        <span style="font-size:0.75rem;color:var(--text-secondary);font-weight:500;line-height:1.2;">${fullName}</span>
                    </div>`;
                } else {
                    predictionBadge = `<span class="prediction-cell-badge dangerous" title="${res.error}">LỖI</span>`;
                }

                let allProbsHtml = '';
                if (res.success && res.all_predictions) {
                    allProbsHtml = `<div class="batch-table-probs-container">`;
                    res.all_predictions.forEach(item => {
                        const pct = item.percentage;
                        let statusClass = 'low-prob';
                        if (pct >= 50) {
                            statusClass = 'high-prob';
                        } else if (pct >= 5) {
                            statusClass = 'medium-prob';
                        }
                        allProbsHtml += `
                            <div class="batch-table-prob-badge ${statusClass}" title="${item.full_name}">
                                <span>${item.class_code.toUpperCase()}:</span>
                                <strong>${pct}%</strong>
                            </div>
                        `;
                    });
                    allProbsHtml += `</div>`;
                } else {
                    allProbsHtml = `<span style="color:var(--text-muted)">-</span>`;
                }

                const imgPreviewHtml = res.image_url 
                    ? `<img src="${res.image_url}" class="batch-table-img-preview" alt="${res.image_id}">` 
                    : '<span style="display:inline-block;width:32px;height:32px;border-radius:6px;background:var(--border-color);margin-right:0.5rem;vertical-align:middle;"></span>';

                tr.innerHTML = `
                    <td>${imgPreviewHtml}<strong>${res.image_id}</strong></td>
                    <td>${res.age !== undefined ? res.age : 'Không rõ'}</td>
                    <td>${translate(res.sex)}</td>
                    <td>${translate(res.localization)}</td>
                    <td>${predictionBadge}</td>
                    <td>${allProbsHtml}</td>
                `;

                batchTableBody.appendChild(tr);
            }

            // 2. Render Gallery Card
            if (batchGridWrapper) {
                const card = document.createElement('div');
                card.className = 'batch-result-card';
                card.setAttribute('data-index', index);
                
                let predictionBadge = '';
                let confidenceText = '';
                let displayDiseaseHtml = '';
                let cardProbsHtml = '';
                if (res.success) {
                    const cls = res.predicted_class.toLowerCase();
                    let badgeClass = 'benign';
                    if (cls === 'mel' || cls === 'bcc') {
                        badgeClass = 'dangerous';
                    } else if (cls === 'akiec') {
                        badgeClass = 'suspicious';
                    }
                    predictionBadge = `<span class="prediction-cell-badge batch-result-card-badge ${badgeClass}">${res.predicted_class.toUpperCase()}</span>`;
                    confidenceText = `${res.confidence || 'Không rõ'}`;
                    const fullName = res.predicted_class_full_name || translateClassCode(cls);
                    displayDiseaseHtml = `<div style="font-size:0.8rem;font-weight:600;color:var(--text-primary);margin-top:0.25rem;margin-bottom:0.5rem;line-height:1.3;">
                        ${fullName}
                    </div>`;

                    if (res.all_predictions) {
                        cardProbsHtml = `<div class="batch-card-probs">`;
                        res.all_predictions.forEach(item => {
                            const pct = item.percentage;
                            let statusClass = 'low-prob';
                            if (pct >= 50) {
                                statusClass = 'high-prob';
                            } else if (pct >= 5) {
                                statusClass = 'medium-prob';
                            }
                            cardProbsHtml += `
                                <div class="batch-card-prob-row ${statusClass}" title="${item.full_name}">
                                    <span class="prob-name">${item.class_code.toUpperCase()}</span>
                                    <div class="prob-bar-track">
                                        <div class="prob-bar-fill" style="width: ${pct}%"></div>
                                    </div>
                                    <span class="prob-pct">${pct}%</span>
                                </div>
                            `;
                        });
                        cardProbsHtml += `</div>`;
                    }
                } else {
                    predictionBadge = `<span class="prediction-cell-badge batch-result-card-badge dangerous" title="${res.error}">LỖI</span>`;
                    confidenceText = 'Thất bại';
                    displayDiseaseHtml = `<div style="font-size:0.8rem;font-weight:600;color:var(--danger-color);margin-top:0.25rem;margin-bottom:0.5rem;">
                        Chẩn đoán lỗi
                    </div>`;
                }

                const age = res.age !== undefined ? res.age : 'Không rõ';
                const sex = translate(res.sex || 'Unknown');
                const loc = translate(res.localization || 'Unknown');
                
                const cardImgHtml = res.image_url 
                    ? `<img src="${res.image_url}" class="batch-result-card-img" alt="${res.image_id}">`
                    : `<div class="batch-result-card-img" style="display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,0.02);color:var(--text-muted);font-size:0.85rem;">Không có ảnh</div>`;
                    
                card.innerHTML = `
                    ${cardImgHtml}
                    <div class="batch-result-card-info">
                        <span class="batch-result-card-id">${res.image_id}</span>
                        <div class="batch-result-card-meta">
                            <span>Tuổi: <strong>${age}</strong></span> • 
                            <span>Giới tính: <strong>${sex}</strong></span>
                        </div>
                        <div class="batch-result-card-meta" style="margin-bottom: 0.5rem;">
                            <span>Vị trí: <strong>${loc}</strong></span>
                        </div>
                        ${displayDiseaseHtml}
                        ${cardProbsHtml}
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:auto;padding-top:0.25rem;border-top:1px solid rgba(255,255,255,0.03);">
                            ${predictionBadge}
                            <span style="font-size:0.8rem;color:var(--text-muted);font-weight:600;">${confidenceText}</span>
                        </div>
                    </div>
                `;
                
                batchGridWrapper.appendChild(card);
            }
        });
    }

    // ==========================================
    // DIAGNOSTIC DETAILS MODAL INTERACTION LOGIC
    // ==========================================
    const detailsModal = document.getElementById('details-modal');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const modalImage = document.getElementById('modal-image');
    const modalImageDimensions = document.getElementById('modal-image-dimensions');
    const modalViewOriginalBtn = document.getElementById('modal-view-original-btn');
    const modalImageId = document.getElementById('modal-image-id');
    const modalTopClassName = document.getElementById('modal-top-class-name');
    const modalTopProbabilityBadge = document.getElementById('modal-top-probability-badge');
    const modalAgeVal = document.getElementById('modal-age-val');
    const modalSexVal = document.getElementById('modal-sex-val');
    const modalLocVal = document.getElementById('modal-loc-val');
    const modalClassDescription = document.getElementById('modal-class-description');
    const modalDistributionBarsContainer = document.getElementById('modal-distribution-bars-container');
    const modalBackdrop = detailsModal ? detailsModal.querySelector('.modal-backdrop') : null;

    function showModalDetails(res) {
        if (!res) return;

        // Reset visibility state
        modalImage.style.display = 'none';
        modalImageDimensions.textContent = '';
        modalViewOriginalBtn.style.display = 'none';

        if (res.success) {
            // Set up image elements
            if (res.image_url) {
                modalImage.src = res.image_url;
                modalImage.style.display = 'block';
                modalViewOriginalBtn.href = res.image_url;
                modalViewOriginalBtn.style.display = 'inline-flex';
                
                // Get image dimension values dynamically
                const tempImg = new Image();
                tempImg.onload = function() {
                    modalImageDimensions.textContent = `${this.naturalWidth} x ${this.naturalHeight}`;
                };
                tempImg.src = res.image_url;
            } else {
                modalImage.src = '';
                modalImageDimensions.textContent = 'Không có ảnh';
            }

            modalImageId.textContent = res.image_id;
            modalTopClassName.textContent = res.predicted_class_full_name;
            modalTopProbabilityBadge.textContent = `${res.confidence} Độ tin cậy`;
            modalTopProbabilityBadge.className = 'probability-badge';
            
            // Set tabular patient info
            modalAgeVal.textContent = res.age !== undefined ? res.age : 'Không rõ';
            modalSexVal.textContent = translate(res.sex);
            modalLocVal.textContent = translate(res.localization);
            
            // Render Description
            const topPrediction = res.all_predictions ? res.all_predictions[0] : null;
            modalClassDescription.textContent = topPrediction ? topPrediction.description : 'Không có mô tả cho bệnh lý này.';

            // Render distribution bars
            modalDistributionBarsContainer.innerHTML = '';
            if (res.all_predictions) {
                res.all_predictions.forEach((item, index) => {
                    const isTop = index === 0;
                    const barItem = document.createElement('div');
                    barItem.className = `bar-item ${isTop ? 'top-match-bar' : ''}`;

                    barItem.innerHTML = `
                        <div class="bar-labels">
                            <span class="bar-name">${item.full_name} <small style="color:var(--text-muted)">(${item.class_code.toUpperCase()})</small></span>
                            <span class="bar-pct ${isTop ? 'top-match-pct' : ''}">${item.percentage}%</span>
                        </div>
                        <div class="bar-track">
                            <div class="bar-fill" data-width="${item.percentage}%"></div>
                        </div>
                    `;
                    modalDistributionBarsContainer.appendChild(barItem);
                });

                // Trigger animations after insertion
                setTimeout(() => {
                    const fills = modalDistributionBarsContainer.querySelectorAll('.bar-fill');
                    fills.forEach(fill => {
                        fill.style.width = fill.getAttribute('data-width');
                    });
                }, 100);
            }
        } else {
            // Render failure details in modal
            modalImage.style.display = 'none';
            modalImageId.textContent = res.image_id;
            modalTopClassName.textContent = 'Không thể phân tích';
            modalTopProbabilityBadge.textContent = 'Thất bại';
            modalTopProbabilityBadge.className = 'probability-badge dangerous-badge';
            
            modalAgeVal.textContent = '-';
            modalSexVal.textContent = '-';
            modalLocVal.textContent = '-';
            
            modalClassDescription.textContent = res.error || 'Đã xảy ra lỗi trong quá trình chẩn đoán mẫu này.';
            modalDistributionBarsContainer.innerHTML = '<p style="color: var(--text-muted); font-size: 0.9rem;">Không có phân phối xác suất do chẩn đoán thất bại.</p>';
        }

        if (detailsModal) {
            detailsModal.style.display = 'flex';
            document.body.classList.add('modal-open');
        }
    }

    function closeModal() {
        if (detailsModal) {
            detailsModal.style.display = 'none';
            document.body.classList.remove('modal-open');
        }
    }

    // Modal Close Triggers
    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', closeModal);
    }
    if (modalBackdrop) {
        modalBackdrop.addEventListener('click', closeModal);
    }
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && detailsModal && detailsModal.style.display === 'flex') {
            closeModal();
        }
    });

    // Click delegation on table rows
    if (batchTableBody) {
        batchTableBody.addEventListener('click', (e) => {
            const tr = e.target.closest('tr');
            if (tr) {
                const idx = tr.getAttribute('data-index');
                if (idx !== null && currentBatchResults[idx]) {
                    showModalDetails(currentBatchResults[idx]);
                }
            }
        });
    }

    // Click delegation on gallery cards
    if (batchGridWrapper) {
        batchGridWrapper.addEventListener('click', (e) => {
            const card = e.target.closest('.batch-result-card');
            if (card) {
                const idx = card.getAttribute('data-index');
                if (idx !== null && currentBatchResults[idx]) {
                    showModalDetails(currentBatchResults[idx]);
                }
            }
        });
    }
});
