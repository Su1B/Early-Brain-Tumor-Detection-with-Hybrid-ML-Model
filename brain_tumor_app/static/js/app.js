document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const removeBtn = document.getElementById('removeBtn');
    const uploadForm = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    const loadingState = document.getElementById('loadingState');
    const emptyState = document.getElementById('emptyState');
    const resultContent = document.getElementById('resultContent');
    
    const predictionBadge = document.getElementById('predictionBadge');
    const confidenceValue = document.getElementById('confidenceValue');
    const meterFill = document.getElementById('meterFill');
    const downloadPdfBtn = document.getElementById('downloadPdfBtn');
    const printBtn = document.getElementById('printBtn');
    const pdfIframe = document.getElementById('pdfIframe');
    
    let currentFile = null;
    let currentPdfUrl = null;

    // Drag and Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        handleFiles(dt.files);
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files && files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                currentFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    dropZone.style.display = 'none';
                    previewContainer.style.display = 'block';
                    analyzeBtn.disabled = false;
                    
                    resultContent.style.display = 'none';
                    emptyState.style.display = 'block';
                }
                reader.readAsDataURL(file);
            } else {
                alert('Please upload an image file format.');
            }
        }
    }

    removeBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.style.display = 'none';
        dropZone.style.display = 'block';
        analyzeBtn.disabled = true;
        resultContent.style.display = 'none';
        emptyState.style.display = 'block';
    });

    // Form Submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!currentFile) return;

        analyzeBtn.disabled = true;
        analyzeBtn.style.display = 'none';
        loadingState.style.display = 'block';
        emptyState.style.display = 'none';
        resultContent.style.display = 'none';
        
        // Mock prediction steps
        const steps = document.querySelectorAll('.loading-steps li');
        let stepIdx = 0;
        const interval = setInterval(() => {
            if(stepIdx < steps.length) {
                steps.forEach(s => s.classList.remove('active'));
                steps[stepIdx].classList.add('active');
                stepIdx++;
            }
        }, 800);

        try {
            const formData = new FormData();
            formData.append('file', currentFile);

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            clearInterval(interval);

            if (data.status === 'success') {
                displayResults(data);
            } else {
                alert('Analysis Error: ' + data.message);
                resetUploadUI();
            }
        } catch (error) {
            clearInterval(interval);
            console.error(error);
            alert('A network error occurred while reaching the AI model.');
            resetUploadUI();
        }
    });

    function displayResults(data) {
        loadingState.style.display = 'none';
        resultContent.style.display = 'block';
        analyzeBtn.style.display = 'block';
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze New Scan';
        
        predictionBadge.textContent = data.prediction;
        predictionBadge.className = 'badge'; 
        
        if (data.prediction.includes('Tumor Detected')) {
            predictionBadge.classList.add('danger');
        } else {
            predictionBadge.classList.add('success');
        }
        
        const confidencePct = (data.confidence * 100).toFixed(1);
        confidenceValue.textContent = confidencePct + '%';
        
        meterFill.style.width = '0%';
        setTimeout(() => {
            meterFill.style.width = confidencePct + '%';
        }, 50);
        
        currentPdfUrl = data.pdf_url;
        downloadPdfBtn.href = currentPdfUrl;
    }

    function resetUploadUI() {
        loadingState.style.display = 'none';
        analyzeBtn.style.display = 'block';
        analyzeBtn.disabled = false;
        emptyState.style.display = 'block';
    }

    printBtn.addEventListener('click', () => {
        if (currentPdfUrl) {
            // Add timestamp query param to prevent caching
            pdfIframe.src = currentPdfUrl + '?t=' + new Date().getTime();
            pdfIframe.onload = function() {
                pdfIframe.contentWindow.focus();
                pdfIframe.contentWindow.print();
            };
        }
    });
});
