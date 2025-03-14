from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Optional, List
import numpy as np
import base64
import cv2
import time
import io

from app.core.face_detector import FaceDetector
from app.models.schemas import (
    FaceDetectionRequest,
    FaceDetectionResponse,
    FaceAlignmentRequest,
    FaceAlignmentResponse,
    FaceExtractionRequest,
    FaceExtractionResponse,
)

router = APIRouter()
face_detector = FaceDetector()

@router.post("/detect", response_model=FaceDetectionResponse)
async def detect_faces(request: FaceDetectionRequest):
    """
    Detect faces in an image.
    
    Parameters:
    - image: Base64 encoded image
    - min_face_size: Minimum face size to detect (default: 50)
    - threshold: Detection confidence threshold (default: 0.85)
    - return_landmarks: Whether to return facial landmarks (default: True)
    
    Returns:
    - faces: List of detected faces with bounding boxes, confidence, and landmarks
    - count: Number of faces detected
    - processing_time_ms: Processing time in milliseconds
    """
    try:
        # Decode base64 image
        start_time = time.time()
        
        # Debug info
        print(f"DEBUG detect_faces - Processing request with image length: {len(request.image)} chars")
        
        try:
            image_data = base64.b64decode(request.image)
            print(f"DEBUG detect_faces - Decoded image data length: {len(image_data)} bytes")
            
            nparr = np.frombuffer(image_data, np.uint8)
            print(f"DEBUG detect_faces - Numpy array shape: {nparr.shape}")
            
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print(f"DEBUG detect_faces - Decoded image shape: {image.shape if image is not None else 'None'}")
        except Exception as e:
            print(f"DEBUG detect_faces - Error decoding image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error decoding image: {str(e)}")
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data or format")
        
        # Detect faces
        faces = face_detector.detect(
            image, 
            min_face_size=request.min_face_size, 
            threshold=request.threshold
        )
        
        # Format response
        result = []
        for face in faces:
            bbox = face['box']
            confidence = face['confidence']
            landmarks = face['keypoints'] if request.return_landmarks else None
            
            face_data = {
                "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                "confidence": float(confidence),
            }
            
            if request.return_landmarks:
                # Convert landmarks to list format
                landmark_list = []
                for key, value in landmarks.items():
                    landmark_list.append([int(value[0]), int(value[1])])
                face_data["landmarks"] = landmark_list
                
            result.append(face_data)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "faces": result,
            "count": len(result),
            "processing_time_ms": processing_time
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"DEBUG detect_faces - Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error detecting faces: {str(e)}")

@router.post("/align", response_model=FaceAlignmentResponse)
async def align_face(request: FaceAlignmentRequest):
    """
    Align a face using facial landmarks.
    
    Parameters:
    - image: Base64 encoded image
    - landmarks: List of 5 facial landmarks (eyes, nose, mouth corners)
        - Format 1 (recommended): [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]
        - Format 2 (legacy): [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
        - Format 3 (legacy): any array that can be converted to 5 landmarks
    - output_size: Size of the output image
    
    Returns:
    - aligned_face: Base64 encoded aligned face image
    - processing_time_ms: Processing time in milliseconds
    """
    try:
        start_time = time.time()
        
        # Debug info
        print(f"DEBUG align_face - Processing request with image length: {len(request.image)} chars")
        print(f"DEBUG align_face - Landmarks: {request.landmarks}")
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
            print(f"DEBUG align_face - Decoded image data length: {len(image_data)} bytes")
            
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print(f"DEBUG align_face - Decoded image shape: {image.shape if image is not None else 'None'}")
        except Exception as e:
            print(f"DEBUG align_face - Error decoding image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error decoding image: {str(e)}")
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data or format")
        
        # Align face
        aligned_face = face_detector.align(image, request.landmarks, request.output_size)
        
        # Encode the aligned face as base64
        try:
            # Check if image is valid
            if aligned_face is None:
                print("DEBUG align_face - Aligned face is None!")
                aligned_face = np.zeros((*request.output_size, 3), dtype=np.uint8)
            
            # Check if image has color channels
            if len(aligned_face.shape) < 3 or aligned_face.shape[2] != 3:
                print(f"DEBUG align_face - Invalid aligned face shape: {aligned_face.shape}")
                # Convert to 3 channels if needed
                if len(aligned_face.shape) == 2:
                    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_GRAY2BGR)
                else:
                    # Create a new image
                    aligned_face = np.zeros((*request.output_size, 3), dtype=np.uint8)
            
            # Check if image has valid values
            if np.max(aligned_face) < 10:
                print("DEBUG align_face - Aligned face is too dark!")
                # Create an error image
                aligned_face = np.ones((*request.output_size, 3), dtype=np.uint8) * 128
                cv2.putText(aligned_face, "Error: Too dark", (10, request.output_size[1]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
            # Encode to JPEG
            success, buffer = cv2.imencode('.jpg', aligned_face)
            if not success:
                print("DEBUG align_face - Error encoding image to JPEG")
                raise ValueError("Failed to encode aligned face to JPEG")
            
            aligned_face_base64 = base64.b64encode(buffer).decode('utf-8')
            print(f"DEBUG align_face - Base64 output length: {len(aligned_face_base64)} chars")
        except Exception as e:
            print(f"DEBUG align_face - Error encoding image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error encoding aligned face: {str(e)}")
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "aligned_face": aligned_face_base64,
            "processing_time_ms": processing_time
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"DEBUG align_face - Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error aligning face: {str(e)}")

@router.post("/extract", response_model=FaceExtractionResponse)
async def extract_face(request: FaceExtractionRequest):
    """
    Extract a face from an image using the bounding box.
    
    Parameters:
    - image: Base64 encoded image
    - bbox: Bounding box coordinates
        - Format 1 (recommended): [x1, y1, x2, y2] (top-left and bottom-right corners)
        - Format 2 (legacy): Any array that can be converted to bbox
    - margin: Margin to add around the bounding box (default: 0.2)
    - output_size: Output image size [width, height] (default: [160, 160])
    
    Returns:
    - face_image: Base64 encoded extracted face image
    - processing_time_ms: Processing time in milliseconds
    """
    try:
        start_time = time.time()
        
        # Debug info
        print(f"DEBUG extract_face - Processing request with image length: {len(request.image)} chars")
        print(f"DEBUG extract_face - Bbox: {request.bbox}")
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
            print(f"DEBUG extract_face - Decoded image data length: {len(image_data)} bytes")
            
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print(f"DEBUG extract_face - Decoded image shape: {image.shape if image is not None else 'None'}")
        except Exception as e:
            print(f"DEBUG extract_face - Error decoding image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error decoding image: {str(e)}")
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data or format")
        
        # Extract face
        bbox = request.bbox  # [x1, y1, x2, y2]
        
        # Convert to the format expected by the extract function [x, y, width, height]
        bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        
        # Print for debugging
        print(f"DEBUG extract_face - Input bbox: {bbox}, Converted bbox: {bbox_xywh}")
        
        face_image = face_detector.extract(image, bbox_xywh, request.margin, request.output_size)
        
        # Encode the face image as base64
        try:
            # Check if image is valid
            if face_image is None:
                print("DEBUG extract_face - Extracted face is None!")
                face_image = np.zeros((*request.output_size, 3), dtype=np.uint8)
            
            # Check if image has color channels
            if len(face_image.shape) < 3 or face_image.shape[2] != 3:
                print(f"DEBUG extract_face - Invalid face image shape: {face_image.shape}")
                # Convert to 3 channels if needed
                if len(face_image.shape) == 2:
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
                else:
                    # Create a new image
                    face_image = np.zeros((*request.output_size, 3), dtype=np.uint8)
            
            # Check if image has valid values
            if np.max(face_image) < 10:
                print("DEBUG extract_face - Extracted face is too dark!")
                # Create an error image
                face_image = np.ones((*request.output_size, 3), dtype=np.uint8) * 64
                cv2.putText(face_image, "Error: Too dark", (10, request.output_size[1]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
            # Encode to JPEG
            success, buffer = cv2.imencode('.jpg', face_image)
            if not success:
                print("DEBUG extract_face - Error encoding image to JPEG")
                raise ValueError("Failed to encode face image to JPEG")
            
            face_image_base64 = base64.b64encode(buffer).decode('utf-8')
            print(f"DEBUG extract_face - Base64 output length: {len(face_image_base64)} chars")
        except Exception as e:
            print(f"DEBUG extract_face - Error encoding image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error encoding extracted face: {str(e)}")
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "face_image": face_image_base64,
            "processing_time_ms": processing_time
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"DEBUG extract_face - Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting face: {str(e)}")

@router.get("/demo")
async def face_detection_demo():
    """
    Show a simple demo page for testing face detection, alignment, and extraction.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Detection Demo</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            .container { display: flex; flex-wrap: wrap; }
            .result-box { margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; width: 300px; }
            img { max-width: 100%; border: 1px solid #eee; }
            textarea { width: 100%; height: 100px; }
            button { padding: 8px 16px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; margin-top: 10px; }
            .log { font-family: monospace; font-size: 12px; white-space: pre-wrap; background: #f5f5f5; padding: 10px; max-height: 200px; overflow: auto; }
        </style>
    </head>
    <body>
        <h1>Face Detection Demo</h1>
        
        <div>
            <h2>Upload Image</h2>
            <input type="file" id="image-upload" accept="image/*">
            <button onclick="detectFace()">Detect Face</button>
        </div>
        
        <div class="container">
            <div class="result-box">
                <h3>Original Image</h3>
                <img id="original-image" src="" alt="Original image will appear here">
            </div>
            
            <div class="result-box">
                <h3>Detected Face</h3>
                <div id="detection-result"></div>
                <canvas id="detection-canvas"></canvas>
            </div>
            
            <div class="result-box">
                <h3>Aligned Face</h3>
                <img id="aligned-face" src="" alt="Aligned face will appear here">
            </div>
            
            <div class="result-box">
                <h3>Extracted Face</h3>
                <img id="extracted-face" src="" alt="Extracted face will appear here">
            </div>
        </div>
        
        <div>
            <h3>Debug Log</h3>
            <div id="debug-log" class="log"></div>
        </div>
        
        <script>
            function log(message) {
                const logElement = document.getElementById('debug-log');
                logElement.innerHTML += message + '\\n';
                logElement.scrollTop = logElement.scrollHeight;
            }
            
            async function detectFace() {
                const fileInput = document.getElementById('image-upload');
                if (!fileInput.files || fileInput.files.length === 0) {
                    alert('Please select an image first');
                    return;
                }
                
                const file = fileInput.files[0];
                const reader = new FileReader();
                
                reader.onload = async function(e) {
                    // Display original image
                    document.getElementById('original-image').src = e.target.result;
                    
                    // Get base64 data (remove data URL prefix)
                    const base64Image = e.target.result.split(',')[1];
                    log('Image loaded: ' + file.name + ' (' + base64Image.length + ' chars)');
                    
                    try {
                        // Step 1: Detect face
                        log('Detecting faces...');
                        const detectResponse = await fetch('/v1/face/detect', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                image: base64Image,
                                min_face_size: 50,
                                threshold: 0.85,
                                return_landmarks: true
                            })
                        });
                        
                        if (!detectResponse.ok) {
                            const errorText = await detectResponse.text();
                            throw new Error('Face detection failed: ' + errorText);
                        }
                        
                        const detectData = await detectResponse.json();
                        log('Detection result: ' + JSON.stringify(detectData));
                        
                        // Display detection result
                        document.getElementById('detection-result').innerHTML = 
                            `<p>Found ${detectData.count} face(s)</p>`;
                        
                        // Draw bounding boxes on canvas
                        if (detectData.count > 0) {
                            const face = detectData.faces[0];
                            log('Selected face: ' + JSON.stringify(face));
                            
                            // Draw on canvas
                            const img = document.getElementById('original-image');
                            const canvas = document.getElementById('detection-canvas');
                            canvas.width = img.naturalWidth;
                            canvas.height = img.naturalHeight;
                            const ctx = canvas.getContext('2d');
                            
                            // Draw original image
                            ctx.drawImage(img, 0, 0);
                            
                            // Draw bounding box
                            const bbox = face.bbox;
                            ctx.strokeStyle = 'red';
                            ctx.lineWidth = 2;
                            ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
                            
                            // Draw landmarks
                            if (face.landmarks) {
                                ctx.fillStyle = 'blue';
                                for (const point of face.landmarks) {
                                    ctx.beginPath();
                                    ctx.arc(point[0], point[1], 3, 0, 2 * Math.PI);
                                    ctx.fill();
                                }
                            }
                            
                            // Step 2: Align face
                            log('Aligning face using landmarks: ' + JSON.stringify(face.landmarks));
                            const alignResponse = await fetch('/v1/face/align', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({
                                    image: base64Image,
                                    landmarks: face.landmarks,
                                    output_size: [160, 160]
                                })
                            });
                            
                            if (!alignResponse.ok) {
                                const errorText = await alignResponse.text();
                                throw new Error('Face alignment failed: ' + errorText);
                            }
                            
                            const alignData = await alignResponse.json();
                            log('Alignment result: processing time = ' + alignData.processing_time_ms + 'ms');
                            
                            // Display aligned face
                            document.getElementById('aligned-face').src = 
                                `data:image/jpeg;base64,${alignData.aligned_face}`;
                            
                            // Step 3: Extract face
                            log('Extracting face using bbox: ' + JSON.stringify(face.bbox));
                            const extractResponse = await fetch('/v1/face/extract', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({
                                    image: base64Image,
                                    bbox: face.bbox,
                                    margin: 0.2,
                                    output_size: [160, 160]
                                })
                            });
                            
                            if (!extractResponse.ok) {
                                const errorText = await extractResponse.text();
                                throw new Error('Face extraction failed: ' + errorText);
                            }
                            
                            const extractData = await extractResponse.json();
                            log('Extraction result: processing time = ' + extractData.processing_time_ms + 'ms');
                            
                            // Display extracted face
                            document.getElementById('extracted-face').src = 
                                `data:image/jpeg;base64,${extractData.face_image}`;
                            
                            log('All processing completed successfully');
                        }
                    } catch (error) {
                        log('ERROR: ' + error.message);
                        console.error('Error:', error);
                        alert(`Error: ${error.message}`);
                    }
                };
                
                reader.readAsDataURL(file);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)