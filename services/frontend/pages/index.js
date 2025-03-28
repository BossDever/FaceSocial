import React, { useState, useRef } from 'react';
import dynamic from 'next/dynamic';
import { Container, Row, Col, Button, Card, Form, Alert, Tabs, Tab } from 'react-bootstrap';
import axios from 'axios';

import 'bootstrap/dist/css/bootstrap.min.css';

// Import Webcam แบบ dynamic เพื่อป้องกันปัญหา SSR
const Webcam = dynamic(() => import('react-webcam'), { ssr: false });

const API_URL = 'http://localhost:8000';

export default function Home() {
  const webcamRef = useRef(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [secondImage, setSecondImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState('recognition');
  
  // สำหรับใช้ปรับแต่งน้ำหนักโมเดล - ปรับให้ไม่มี FaceNet
  const [modelWeights, setModelWeights] = useState({
    arcface: 0.33,
    adaface: 0.33,
    elasticface: 0.34
  });

  const captureImage = () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      setCapturedImage(imageSrc);
    }
  };

  const captureSecondImage = () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      setSecondImage(imageSrc);
    }
  };

  const reset = () => {
    setCapturedImage(null);
    setSecondImage(null);
    setResult(null);
  };

  const handleImageUpload = (event, setImageFunction) => {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onloadend = () => {
      setImageFunction(reader.result);
    };
    if (file) {
      reader.readAsDataURL(file);
    }
  };

  const handleWeightChange = (model, value) => {
    setModelWeights({
      ...modelWeights,
      [model]: parseFloat(value)
    });
  };

  const compareFaces = async () => {
    if (!capturedImage || !secondImage) return;
    
    setIsLoading(true);
    
    try {
      // แปลงรูปภาพจาก Data URL เป็น Blob
      const fetchImage = async (dataUrl) => {
        const response = await fetch(dataUrl);
        const blob = await response.blob();
        return blob;
      };
      
      const image1 = await fetchImage(capturedImage);
      const image2 = await fetchImage(secondImage);
      
      // สร้าง FormData
      const formData = new FormData();
      formData.append('image1', image1);
      formData.append('image2', image2);
      formData.append('model_weights', JSON.stringify(modelWeights));
      
      // ส่งคำขอไปยัง API
      const response = await axios.post(`${API_URL}/api/v1/face-recognition/compare`, formData);
      
      setResult(response.data);
    } catch (error) {
      console.error('Error comparing faces:', error);
      setResult({ error: 'Failed to compare faces' });
    } finally {
      setIsLoading(false);
    }
  };

  const checkSecurity = async () => {
    // โค้ดเดิมสำหรับ checkSecurity...
    if (!capturedImage) return;
    
    setIsLoading(true);
    
    try {
      const fetchImage = async (dataUrl) => {
        const response = await fetch(dataUrl);
        const blob = await response.blob();
        return blob;
      };
      
      const image = await fetchImage(capturedImage);
      
      const formData = new FormData();
      formData.append('image', image);
      formData.append('checks', 'liveness,deepfake,spoofing');
      
      const response = await axios.post(`${API_URL}/api/v1/security/check`, formData);
      
      setResult(response.data);
    } catch (error) {
      console.error('Error checking security:', error);
      setResult({ error: 'Failed to check security' });
    } finally {
      setIsLoading(false);
    }
  };

  const detectFaces = async () => {
    // โค้ดเดิมสำหรับ detectFaces...
    if (!capturedImage) return;
    
    setIsLoading(true);
    
    try {
      const fetchImage = async (dataUrl) => {
        const response = await fetch(dataUrl);
        const blob = await response.blob();
        return blob;
      };
      
      const image = await fetchImage(capturedImage);
      
      const formData = new FormData();
      formData.append('image', image);
      
      const response = await axios.post(`${API_URL}/api/v1/face-detection`, formData);
      
      setResult(response.data);
    } catch (error) {
      console.error('Error detecting faces:', error);
      setResult({ error: 'Failed to detect faces' });
    } finally {
      setIsLoading(false);
    }
  };

  const analyzeImage = () => {
    if (activeTab === 'recognition') {
      compareFaces();
    } else if (activeTab === 'security') {
      checkSecurity();
    } else if (activeTab === 'detection') {
      detectFaces();
    }
  };

  return (
    <Container>
      <h1 className="text-center mb-4">FaceSocial AI</h1>
      
      <Tabs
        activeKey={activeTab}
        onSelect={(k) => setActiveTab(k)}
        className="mb-3"
      >
        <Tab eventKey="recognition" title="Face Recognition">
          {/* ส่วนโค้ด UI ตามเดิม */}
          <Row>
            <Col md={6}>
              <Card className="mb-3">
                <Card.Body>
                  <h5>First Image</h5>
                  {capturedImage ? (
                    <img src={capturedImage} alt="Captured" className="img-fluid rounded" />
                  ) : (
                    <div className="webcam-container">
                      <Webcam
                        audio={false}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        className="webcam"
                      />
                    </div>
                  )}
                  
                  <div className="mt-3">
                    {capturedImage ? (
                      <Button variant="outline-secondary" onClick={() => setCapturedImage(null)}>
                        Retake
                      </Button>
                    ) : (
                      <Button variant="primary" onClick={captureImage}>
                        Capture
                      </Button>
                    )}
                    <Form.Group className="mt-2">
                      <Form.Label>Or upload an image</Form.Label>
                      <Form.Control 
                        type="file" 
                        accept="image/*" 
                        onChange={(e) => handleImageUpload(e, setCapturedImage)} 
                      />
                    </Form.Group>
                  </div>
                </Card.Body>
              </Card>
            </Col>
            
            <Col md={6}>
              <Card className="mb-3">
                <Card.Body>
                  <h5>Second Image</h5>
                  {secondImage ? (
                    <img src={secondImage} alt="Second" className="img-fluid rounded" />
                  ) : (
                    <div className="webcam-container">
                      <Webcam
                        audio={false}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        className="webcam"
                      />
                    </div>
                  )}
                  
                  <div className="mt-3">
                    {secondImage ? (
                      <Button variant="outline-secondary" onClick={() => setSecondImage(null)}>
                        Retake
                      </Button>
                    ) : (
                      <Button variant="primary" onClick={captureSecondImage}>
                        Capture
                      </Button>
                    )}
                    <Form.Group className="mt-2">
                      <Form.Label>Or upload an image</Form.Label>
                      <Form.Control 
                        type="file" 
                        accept="image/*" 
                        onChange={(e) => handleImageUpload(e, setSecondImage)} 
                      />
                    </Form.Group>
                  </div>
                </Card.Body>
              </Card>
            </Col>
          </Row>
          
          <div className="d-flex justify-content-center mt-3 mb-4">
            <Button 
              variant="success" 
              onClick={analyzeImage}
              disabled={isLoading || !capturedImage || !secondImage}
              className="me-2"
            >
              {isLoading ? 'Processing...' : 'Compare Faces'}
            </Button>
            <Button 
              variant="secondary" 
              onClick={reset}
              disabled={isLoading}
            >
              Reset
            </Button>
          </div>
          
          {result && !result.error && (
            <Card className="result-container">
              <Card.Body>
                <h4>Comparison Result</h4>
                <Alert variant={result.is_match ? 'success' : 'danger'}>
                  {result.is_match ? 'The faces match!' : 'The faces do not match!'}
                </Alert>
                <p><strong>Similarity Score:</strong> {(result.similarity * 100).toFixed(2)}%</p>
                
                <h5 className="mt-3">Model Details</h5>
                {result.model_details && Object.entries(result.model_details).map(([model, score]) => (
                  <div key={model} className="mb-2">
                    <strong>{model}:</strong> {(score * 100).toFixed(2)}%
                  </div>
                ))}
              </Card.Body>
            </Card>
          )}
          
          <Card className="settings mt-4">
            <Card.Body>
              <h4>Advanced Settings</h4>
              <p>Adjust model weights for face recognition:</p>
              
              <div className="model-weights">
                {Object.entries(modelWeights).map(([model, weight]) => (
                  <Form.Group key={model} className="mb-3">
                    <Form.Label>{model}</Form.Label>
                    <Form.Control 
                      type="range" 
                      min="0" 
                      max="1" 
                      step="0.05" 
                      value={weight}
                      onChange={(e) => handleWeightChange(model, e.target.value)}
                    />
                    <div className="d-flex justify-content-between">
                      <small>0</small>
                      <small>{weight.toFixed(2)}</small>
                      <small>1</small>
                    </div>
                  </Form.Group>
                ))}
              </div>
            </Card.Body>
          </Card>
        </Tab>
        
        <Tab eventKey="security" title="Liveness & Deepfake Detection">
          {/* ส่วนโค้ด Security Tab */}
          <Row>
            <Col md={6}>
              <Card className="mb-3">
                <Card.Body>
                  <h5>Capture Image</h5>
                  {capturedImage ? (
                    <img src={capturedImage} alt="Captured" className="img-fluid rounded" />
                  ) : (
                    <div className="webcam-container">
                      <Webcam
                        audio={false}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        className="webcam"
                      />
                    </div>
                  )}
                  
                  <div className="mt-3">
                    {capturedImage ? (
                      <Button variant="outline-secondary" onClick={() => setCapturedImage(null)}>
                        Retake
                      </Button>
                    ) : (
                      <Button variant="primary" onClick={captureImage}>
                        Capture
                      </Button>
                    )}
                    <Form.Group className="mt-2">
                      <Form.Label>Or upload an image</Form.Label>
                      <Form.Control 
                        type="file" 
                        accept="image/*" 
                        onChange={(e) => handleImageUpload(e, setCapturedImage)} 
                      />
                    </Form.Group>
                  </div>
                </Card.Body>
              </Card>
            </Col>
            
            <Col md={6}>
              <Card className="mb-3">
                <Card.Body>
                  <h5>Security Checks</h5>
                  <p>Select which security checks to perform:</p>
                  <Form.Check 
                    type="checkbox" 
                    label="Liveness Detection" 
                    defaultChecked 
                  />
                  <Form.Check 
                    type="checkbox" 
                    label="Deepfake Detection" 
                    defaultChecked 
                  />
                  <Form.Check 
                    type="checkbox" 
                    label="Spoofing Detection" 
                    defaultChecked 
                  />
                  
                  <div className="mt-3">
                    <Button 
                      variant="success" 
                      onClick={analyzeImage}
                      disabled={isLoading || !capturedImage}
                      className="me-2"
                    >
                      {isLoading ? 'Processing...' : 'Run Security Checks'}
                    </Button>
                    <Button 
                      variant="secondary" 
                      onClick={reset}
                      disabled={isLoading}
                    >
                      Reset
                    </Button>
                  </div>
                </Card.Body>
              </Card>
            </Col>
          </Row>
          
          {result && !result.error && (
            <Card className="result-container">
              <Card.Body>
                <h4>Security Analysis Result</h4>
                <Alert variant={result.is_real_face ? 'success' : 'danger'}>
                  {result.is_real_face ? 'This appears to be a real face!' : 'This does NOT appear to be a real face!'}
                </Alert>
                
                {result.liveness && (
                  <div className="mb-3">
                    <h5>Liveness Check</h5>
                    <p>
                      <strong>Result:</strong> {result.liveness.is_live ? 'Live Face' : 'Not a Live Face'}
                    </p>
                    <p>
                      <strong>Confidence:</strong> {(result.liveness.score * 100).toFixed(2)}%
                    </p>
                  </div>
                )}
                
                {result.deepfake && (
                  <div className="mb-3">
                    <h5>Deepfake Check</h5>
                    <p>
                      <strong>Result:</strong> {result.deepfake.is_fake ? 'Likely Deepfake' : 'Not a Deepfake'}
                    </p>
                    <p>
                      <strong>Confidence:</strong> {(result.deepfake.score * 100).toFixed(2)}%
                    </p>
                  </div>
                )}
                
                {result.spoofing && (
                  <div className="mb-3">
                    <h5>Spoofing Check</h5>
                    <p>
                      <strong>Result:</strong> {result.spoofing.is_attack ? 'Likely Spoofing Attack' : 'Not a Spoofing Attack'}
                    </p>
                    <p>
                      <strong>Confidence:</strong> {(result.spoofing.score * 100).toFixed(2)}%
                    </p>
                  </div>
                )}
              </Card.Body>
            </Card>
          )}
        </Tab>
        
        <Tab eventKey="detection" title="Face Detection">
          {/* ส่วนโค้ด Detection Tab */}
          <Row>
            <Col md={12}>
              <Card className="mb-3">
                <Card.Body>
                  <h5>Capture Image</h5>
                  {capturedImage ? (
                    <img src={capturedImage} alt="Captured" className="img-fluid rounded" />
                  ) : (
                    <div className="webcam-container">
                      <Webcam
                        audio={false}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        className="webcam"
                      />
                    </div>
                  )}
                  
                  <div className="mt-3">
                    {capturedImage ? (
                      <Button variant="outline-secondary" onClick={() => setCapturedImage(null)}>
                        Retake
                      </Button>
                    ) : (
                      <Button variant="primary" onClick={captureImage}>
                        Capture
                      </Button>
                    )}
                    <Form.Group className="mt-2">
                      <Form.Label>Or upload an image</Form.Label>
                      <Form.Control 
                        type="file" 
                        accept="image/*" 
                        onChange={(e) => handleImageUpload(e, setCapturedImage)} 
                      />
                    </Form.Group>
                    
                    <div className="mt-3">
                      <Button 
                        variant="success" 
                        onClick={analyzeImage}
                        disabled={isLoading || !capturedImage}
                        className="me-2"
                      >
                        {isLoading ? 'Processing...' : 'Detect Faces'}
                      </Button>
                      <Button 
                        variant="secondary" 
                        onClick={reset}
                        disabled={isLoading}
                      >
                        Reset
                      </Button>
                    </div>
                  </div>
                </Card.Body>
              </Card>
            </Col>
          </Row>
          
          {result && !result.error && result.faces && (
            <Card className="result-container">
              <Card.Body>
                <h4>Face Detection Results</h4>
                <p>Found {result.faces.length} faces in the image.</p>
                
                <div className="face-grid">
                  {result.faces.map((face, index) => (
                    <div key={index} className="face-card">
                      <img 
                        src={capturedImage} 
                        alt={`Face ${index+1}`}
                        className="face-image"
                      />
                      <p className="mt-2 mb-0">Face #{index+1}</p>
                      <small>Confidence: {(face.confidence * 100).toFixed(1)}%</small>
                    </div>
                  ))}
                </div>
              </Card.Body>
            </Card>
          )}
        </Tab>
      </Tabs>
      
      {result && result.error && (
        <Alert variant="danger">
          Error: {result.error}
        </Alert>
      )}

      <style jsx global>{`
        body {
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
            'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
            sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
          background-color: #f8f9fa;
        }

        .container {
          padding-top: 2rem;
        }

        .webcam-container {
          position: relative;
          width: 100%;
          max-width: 640px;
          margin: 0 auto;
        }

        .webcam {
          width: 100%;
          height: auto;
          border-radius: 8px;
        }

        .controls {
          margin-top: 1rem;
          display: flex;
          gap: 1rem;
          justify-content: center;
        }

        .result-container {
          margin-top: 2rem;
          padding: 1rem;
          border-radius: 8px;
          background-color: white;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .face-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
          gap: 1rem;
          margin-top: 1rem;
        }

        .face-card {
          border: 1px solid #ddd;
          border-radius: 4px;
          padding: 0.5rem;
          text-align: center;
        }

        .face-image {
          width: 100%;
          height: auto;
          border-radius: 4px;
        }

        .settings {
          margin-top: 2rem;
        }

        .model-weights {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 1rem;
          margin-top: 1rem;
        }
      `}</style>
    </Container>
  );
}
