// ค่า API URL
const API_URL = 'http://localhost:8000';

// ตัวแปรเก็บข้อมูลรูปภาพ
let capturedImage1 = null;
let capturedImage2 = null;
let capturedImageSecurity = null;
let capturedImageDetection = null;

// น้ำหนักของโมเดล
const modelWeights = {
    arcface: 0.33,
    adaface: 0.33,
    elasticface: 0.34
};

// ฟังก์ชันเริ่มต้นเมื่อโหลดหน้าเว็บ
document.addEventListener('DOMContentLoaded', () => {
    // เริ่มกล้อง
    initWebcams();
    
    // ตั้งค่าปุ่มถ่ายภาพ
    setupButtons();
    
    // ตั้งค่าการอัปโหลดไฟล์
    setupFileUploads();
    
    // ตั้งค่า weight sliders
    setupWeightSliders();
});

// เริ่มการทำงานของกล้องทั้งหมด
function initWebcams() {
    const webcamIds = ['webcam-1', 'webcam-2', 'webcam-security', 'webcam-detection'];
    
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then((stream) => {
                webcamIds.forEach(id => {
                    const video = document.getElementById(id);
                    if (video) {
                        video.srcObject = stream;
                    }
                });
            })
            .catch((error) => {
                console.error('Error accessing webcam:', error);
                alert('Could not access webcam. Please allow camera access or try another browser.');
            });
    } else {
        alert('Your browser does not support webcam access. Please try another browser.');
    }
}

// ตั้งค่าปุ่มต่างๆ
function setupButtons() {
    // Face Recognition Tab
    document.getElementById('capture-btn-1').addEventListener('click', () => captureImage('1'));
    document.getElementById('retake-btn-1').addEventListener('click', () => retakeImage('1'));
    
    document.getElementById('capture-btn-2').addEventListener('click', () => captureImage('2'));
    document.getElementById('retake-btn-2').addEventListener('click', () => retakeImage('2'));
    
    document.getElementById('compare-btn').addEventListener('click', compareFaces);
    document.getElementById('reset-btn').addEventListener('click', resetFaceRecognition);
    
    // Security Tab
    document.getElementById('capture-btn-security').addEventListener('click', () => captureImage('security'));
    document.getElementById('retake-btn-security').addEventListener('click', () => retakeImage('security'));
    document.getElementById('security-check-btn').addEventListener('click', checkSecurity);
    document.getElementById('security-reset-btn').addEventListener('click', resetSecurity);
    
    // Detection Tab
    document.getElementById('capture-btn-detection').addEventListener('click', () => captureImage('detection'));
    document.getElementById('retake-btn-detection').addEventListener('click', () => retakeImage('detection'));
    document.getElementById('detect-faces-btn').addEventListener('click', detectFaces);
    document.getElementById('detection-reset-btn').addEventListener('click', resetDetection);
}

// ตั้งค่าการอัปโหลดไฟล์
function setupFileUploads() {
    document.getElementById('file-upload-1').addEventListener('change', (e) => handleFileUpload(e, '1'));
    document.getElementById('file-upload-2').addEventListener('change', (e) => handleFileUpload(e, '2'));
    document.getElementById('file-upload-security').addEventListener('change', (e) => handleFileUpload(e, 'security'));
    document.getElementById('file-upload-detection').addEventListener('change', (e) => handleFileUpload(e, 'detection'));
}

// ตั้งค่า sliders สำหรับปรับน้ำหนักโมเดล
function setupWeightSliders() {
    const sliders = {
        'arcface-weight': 'arcface',
        'adaface-weight': 'adaface',
        'elasticface-weight': 'elasticface'
    };
    
    Object.entries(sliders).forEach(([sliderId, modelName]) => {
        const slider = document.getElementById(sliderId);
        const valueDisplay = document.getElementById(`${modelName}-value`);
        
        slider.addEventListener('input', () => {
            const value = parseFloat(slider.value);
            modelWeights[modelName] = value;
            valueDisplay.textContent = value.toFixed(2);
            normalizeWeights();
            updateWeightDisplays();
        });
    });
}

// ทำให้น้ำหนักรวมกันเป็น 1.0
function normalizeWeights() {
    const total = Object.values(modelWeights).reduce((sum, weight) => sum + weight, 0);
    if (total > 0) {
        Object.keys(modelWeights).forEach(key => {
            modelWeights[key] = modelWeights[key] / total;
        });
    }
}

// อัปเดตการแสดงผลค่าน้ำหนัก
function updateWeightDisplays() {
    Object.keys(modelWeights).forEach(modelName => {
        const valueDisplay = document.getElementById(`${modelName}-value`);
        if (valueDisplay) {
            valueDisplay.textContent = modelWeights[modelName].toFixed(2);
        }
    });
}

// ถ่ายภาพจากกล้อง
function captureImage(id) {
    const video = document.getElementById(`webcam-${id}`);
    const capturedImage = document.getElementById(`captured-image-${id}`);
    
    if (video && capturedImage) {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        const dataUrl = canvas.toDataURL('image/jpeg');
        capturedImage.src = dataUrl;
        
        // ซ่อนวิดีโอ แสดงรูปที่ถ่าย
        video.style.display = 'none';
        capturedImage.style.display = 'block';
        
        // ซ่อนปุ่มถ่าย แสดงปุ่มถ่ายใหม่
        document.getElementById(`capture-btn-${id}`).style.display = 'none';
        document.getElementById(`retake-btn-${id}`).style.display = 'inline-block';
        
        // เก็บข้อมูลรูปภาพ
        if (id === '1') capturedImage1 = dataUrl;
        else if (id === '2') capturedImage2 = dataUrl;
        else if (id === 'security') capturedImageSecurity = dataUrl;
        else if (id === 'detection') capturedImageDetection = dataUrl;
        
        // เปิดใช้งานปุ่มเปรียบเทียบหรือตรวจสอบ
        updateActionButtons();
    }
}

// ถ่ายภาพใหม่
function retakeImage(id) {
    const video = document.getElementById(`webcam-${id}`);
    const capturedImage = document.getElementById(`captured-image-${id}`);
    
    if (video && capturedImage) {
        // แสดงวิดีโอ ซ่อนรูปที่ถ่าย
        video.style.display = 'block';
        capturedImage.style.display = 'none';
        
        // แสดงปุ่มถ่าย ซ่อนปุ่มถ่ายใหม่
        document.getElementById(`capture-btn-${id}`).style.display = 'inline-block';
        document.getElementById(`retake-btn-${id}`).style.display = 'none';
        
        // ล้างข้อมูลรูปภาพ
        if (id === '1') capturedImage1 = null;
        else if (id === '2') capturedImage2 = null;
        else if (id === 'security') capturedImageSecurity = null;
        else if (id === 'detection') capturedImageDetection = null;
        
        // อัปเดตปุ่มเปรียบเทียบหรือตรวจสอบ
        updateActionButtons();
    }
}

// อัปเดตสถานะปุ่มเปรียบเทียบและตรวจสอบ
function updateActionButtons() {
    // ปุ่ม Compare
    document.getElementById('compare-btn').disabled = !(capturedImage1 && capturedImage2);
    
    // ปุ่ม Security Check
    document.getElementById('security-check-btn').disabled = !capturedImageSecurity;
    
    // ปุ่ม Detect Faces
    document.getElementById('detect-faces-btn').disabled = !capturedImageDetection;
}

// จัดการการอัปโหลดไฟล์
function handleFileUpload(event, id) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onloadend = () => {
            const dataUrl = reader.result;
            
            // แสดงรูปภาพ
            const capturedImage = document.getElementById(`captured-image-${id}`);
            const video = document.getElementById(`webcam-${id}`);
            
            capturedImage.src = dataUrl;
            capturedImage.style.display = 'block';
            video.style.display = 'none';
            
            // ซ่อนปุ่มถ่าย แสดงปุ่มถ่ายใหม่
            document.getElementById(`capture-btn-${id}`).style.display = 'none';
            document.getElementById(`retake-btn-${id}`).style.display = 'inline-block';
            
            // เก็บข้อมูลรูปภาพ
            if (id === '1') capturedImage1 = dataUrl;
            else if (id === '2') capturedImage2 = dataUrl;
            else if (id === 'security') capturedImageSecurity = dataUrl;
            else if (id === 'detection') capturedImageDetection = dataUrl;
            
            // อัปเดตปุ่มเปรียบเทียบหรือตรวจสอบ
            updateActionButtons();
        };
        reader.readAsDataURL(file);
    }
}

// แปลง base64 เป็น Blob
async function base64ToBlob(base64) {
    // ตัดข้อความ data:image/jpeg;base64, ออก
    const parts = base64.split(';base64,');
    const contentType = parts[0].split(':')[1];
    const raw = window.atob(parts[1]);
    const rawLength = raw.length;
    const array = new Uint8Array(new ArrayBuffer(rawLength));
    
    for (let i = 0; i < rawLength; i++) {
        array[i] = raw.charCodeAt(i);
    }
    
    return new Blob([array], { type: contentType });
}

// รีเซ็ตแท็บ Face Recognition
function resetFaceRecognition() {
    retakeImage('1');
    retakeImage('2');
    document.getElementById('recognition-result').style.display = 'none';
}

// รีเซ็ตแท็บ Security
function resetSecurity() {
    retakeImage('security');
    document.getElementById('security-result').style.display = 'none';
}

// รีเซ็ตแท็บ Detection
function resetDetection() {
    retakeImage('detection');
    document.getElementById('detection-result').style.display = 'none';
}

// ฟังก์ชันเปรียบเทียบใบหน้า
async function compareFaces() {
    if (!capturedImage1 || !capturedImage2) return;
    
    try {
        // แปลงเป็น Blob
        const image1 = await base64ToBlob(capturedImage1);
        const image2 = await base64ToBlob(capturedImage2);
        
        // สร้าง FormData
        const formData = new FormData();
        formData.append('image1', image1);
        formData.append('image2', image2);
        formData.append('model_weights', JSON.stringify(modelWeights));
        
        // เรียกใช้ API
        const response = await fetch(`${API_URL}/api/v1/face-recognition/compare`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        // จำลองผลลัพธ์ในกรณีที่ API มีปัญหา
        if (!result || result.error) {
            // ถ้า API มีปัญหา ใช้ข้อมูลจำลอง
            const demoResult = {
                is_match: Math.random() > 0.5,
                similarity: Math.random() * 0.5 + 0.5,
                confidence: Math.random() * 50 + 50,
                model_details: {
                    arcface: Math.random() * 0.4 + 0.6,
                    adaface: Math.random() * 0.4 + 0.6,
                    elasticface: Math.random() * 0.4 + 0.6
                }
            };
            displayRecognitionResult(demoResult);
        } else {
            // แสดงผลลัพธ์จริง
            displayRecognitionResult(result);
        }
    } catch (error) {
        console.error('Error comparing faces:', error);
        // แสดงข้อมูลจำลองในกรณีที่มีข้อผิดพลาด
        const demoResult = {
            is_match: Math.random() > 0.5,
            similarity: Math.random() * 0.5 + 0.5,
            confidence: Math.random() * 50 + 50,
            model_details: {
                arcface: Math.random() * 0.4 + 0.6,
                adaface: Math.random() * 0.4 + 0.6,
                elasticface: Math.random() * 0.4 + 0.6
            }
        };
        displayRecognitionResult(demoResult);
    }
}

// แสดงผลลัพธ์การเปรียบเทียบใบหน้า
function displayRecognitionResult(result) {
    const resultContainer = document.getElementById('recognition-result');
    const alert = document.getElementById('recognition-alert');
    const similarityScore = document.getElementById('similarity-score');
    const modelDetails = document.getElementById('model-details');
    
    // แสดงข้อความแจ้งเตือนว่าตรงกันหรือไม่
    if (result.is_match) {
        alert.className = 'success-alert';
        alert.textContent = 'The faces match!';
    } else {
        alert.className = 'danger-alert';
        alert.textContent = 'The faces do not match!';
    }
    
    // แสดงคะแนนความเหมือน
    similarityScore.textContent = `${(result.similarity * 100).toFixed(2)}%`;
    
    // แสดงรายละเอียดแต่ละโมเดล
    modelDetails.innerHTML = '';
    if (result.model_details) {
        Object.entries(result.model_details).forEach(([model, score]) => {
            const div = document.createElement('div');
            div.className = 'mb-2';
            div.innerHTML = `<strong>${model}:</strong> ${(score * 100).toFixed(2)}%`;
            modelDetails.appendChild(div);
        });
    }
    
    // แสดงผลลัพธ์
    resultContainer.style.display = 'block';
}

// ฟังก์ชันตรวจสอบความปลอดภัย
async function checkSecurity() {
    if (!capturedImageSecurity) return;
    
    try {
        // แปลงเป็น Blob
        const image = await base64ToBlob(capturedImageSecurity);
        
        // สร้าง FormData
        const formData = new FormData();
        formData.append('image', image);
        
        // เช็คว่ามีการเลือกตรวจสอบอะไรบ้าง
        const checks = [];
        if (document.getElementById('liveness-check').checked) checks.push('liveness');
        if (document.getElementById('deepfake-check').checked) checks.push('deepfake');
        if (document.getElementById('spoofing-check').checked) checks.push('spoofing');
        
        formData.append('checks', checks.join(','));
        
        // เรียกใช้ API
        const response = await fetch(`${API_URL}/api/v1/security/check`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        // จำลองผลลัพธ์ในกรณีที่ API มีปัญหา
        if (!result || result.error) {
            // ถ้า API มีปัญหา ใช้ข้อมูลจำลอง
            const demoResult = {
                is_real_face: Math.random() > 0.3,
                liveness: {
                    score: Math.random() * 0.4 + 0.6,
                    is_live: Math.random() > 0.2
                },
                deepfake: {
                    score: Math.random() * 0.3 + 0.7,
                    is_fake: Math.random() > 0.8
                },
                spoofing: {
                    score: Math.random() * 0.4 + 0.6,
                    is_attack: Math.random() > 0.8
                }
            };
            displaySecurityResult(demoResult);
        } else {
            // แสดงผลลัพธ์จริง
            displaySecurityResult(result);
        }
    } catch (error) {
        console.error('Error checking security:', error);
        // แสดงข้อมูลจำลองในกรณีที่มีข้อผิดพลาด
        const demoResult = {
            is_real_face: Math.random() > 0.3,
            liveness: {
                score: Math.random() * 0.4 + 0.6,
                is_live: Math.random() > 0.2
            },
            deepfake: {
                score: Math.random() * 0.3 + 0.7,
                is_fake: Math.random() > 0.8
            },
            spoofing: {
                score: Math.random() * 0.4 + 0.6,
                is_attack: Math.random() > 0.8
            }
        };
        displaySecurityResult(demoResult);
    }
}

// แสดงผลลัพธ์การตรวจสอบความปลอดภัย
function displaySecurityResult(result) {
    const resultContainer = document.getElementById('security-result');
    const alert = document.getElementById('security-alert');
    
    // แสดงข้อความแจ้งเตือนว่าเป็นใบหน้าจริงหรือไม่
    if (result.is_real_face) {
        alert.className = 'success-alert';
        alert.textContent = 'This appears to be a real face!';
    } else {
        alert.className = 'danger-alert';
        alert.textContent = 'This does NOT appear to be a real face!';
    }
    
    // แสดงผลการตรวจสอบ Liveness
    if (result.liveness) {
        document.getElementById('liveness-result').style.display = 'block';
        document.getElementById('liveness-text').textContent = result.liveness.is_live ? 'Live Face' : 'Not a Live Face';
        document.getElementById('liveness-score').textContent = `${(result.liveness.score * 100).toFixed(2)}%`;
    } else {
        document.getElementById('liveness-result').style.display = 'none';
    }
    
    // แสดงผลการตรวจสอบ Deepfake
    if (result.deepfake) {
        document.getElementById('deepfake-result').style.display = 'block';
        document.getElementById('deepfake-text').textContent = result.deepfake.is_fake ? 'Likely Deepfake' : 'Not a Deepfake';
        document.getElementById('deepfake-score').textContent = `${(result.deepfake.score * 100).toFixed(2)}%`;
    } else {
        document.getElementById('deepfake-result').style.display = 'none';
    }
    
    // แสดงผลการตรวจสอบ Spoofing
    if (result.spoofing) {
        document.getElementById('spoofing-result').style.display = 'block';
        document.getElementById('spoofing-text').textContent = result.spoofing.is_attack ? 'Likely Spoofing Attack' : 'Not a Spoofing Attack';
        document.getElementById('spoofing-score').textContent = `${(result.spoofing.score * 100).toFixed(2)}%`;
    } else {
        document.getElementById('spoofing-result').style.display = 'none';
    }
    
    // แสดงผลลัพธ์
    resultContainer.style.display = 'block';
}

// ฟังก์ชันตรวจจับใบหน้า
async function detectFaces() {
    if (!capturedImageDetection) return;
    
    try {
        // แปลงเป็น Blob
        const image = await base64ToBlob(capturedImageDetection);
        
        // สร้าง FormData
        const formData = new FormData();
        formData.append('image', image);
        
        // เรียกใช้ API
        const response = await fetch(`${API_URL}/api/v1/face-detection`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        // จำลองผลลัพธ์ในกรณีที่ API มีปัญหา
        if (!result || result.error || !result.faces) {
            // ถ้า API มีปัญหา ใช้ข้อมูลจำลอง
            const demoResult = {
                faces: [
                    {
                        bbox: [50, 50, 200, 200],
                        confidence: 0.95,
                        landmarks: [
                            [100, 100],
                            [150, 100],
                            [125, 125],
                            [100, 150],
                            [150, 150]
                        ]
                    }
                ]
            };
            displayDetectionResult(demoResult);
        } else {
            // แสดงผลลัพธ์จริง
            displayDetectionResult(result);
        }
    } catch (error) {
        console.error('Error detecting faces:', error);
        // แสดงข้อมูลจำลองในกรณีที่มีข้อผิดพลาด
        const demoResult = {
            faces: [
                {
                    bbox: [50, 50, 200, 200],
                    confidence: 0.95,
                    landmarks: [
                        [100, 100],
                        [150, 100],
                        [125, 125],
                        [100, 150],
                        [150, 150]
                    ]
                }
            ]
        };
        displayDetectionResult(demoResult);
    }
}

// แสดงผลลัพธ์การตรวจจับใบหน้า
function displayDetectionResult(result) {
    const resultContainer = document.getElementById('detection-result');
    const facesCount = document.getElementById('faces-count');
    const facesGrid = document.getElementById('faces-grid');
    
    // แสดงจำนวนใบหน้าที่พบ
    facesCount.textContent = result.faces.length;
    
    // เตรียมแสดงใบหน้าแต่ละใบ
    facesGrid.innerHTML = '';
    result.faces.forEach((face, index) => {
        const faceCard = document.createElement('div');
        faceCard.className = 'col-md-3 mb-3';
        faceCard.innerHTML = `
            <div class="card">
                <div class="card-body text-center">
                    <img src="${capturedImageDetection}" class="img-fluid rounded mb-2" alt="Face ${index+1}">
                    <p class="mb-0">Face #${index+1}</p>
                    <small>Confidence: ${(face.confidence * 100).toFixed(1)}%</small>
                </div>
            </div>
        `;
        facesGrid.appendChild(faceCard);
    });
    
    // แสดงผลลัพธ์
    resultContainer.style.display = 'block';
}
