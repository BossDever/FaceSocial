import React, { useState, useRef, useEffect } from "react";
import {
  Container,
  Row,
  Col,
  Button,
  Card,
  Form,
  Alert,
  Tabs,
  Tab,
  Table,
  Badge,
} from "react-bootstrap";
import Webcam from "react-webcam";
import axios from "axios";
import RealtimeFaceAnalysis from "./components/RealtimeFaceAnalysis";

const API_URL = "http://localhost:8000";

function App() {
  const webcamRef = useRef(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [secondImage, setSecondImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState("recognition");

  // สำหรับใช้ปรับแต่งน้ำหนักโมเดล - ปรับให้ไม่มี FaceNet
  const [modelWeights, setModelWeights] = useState({
    arcface: 0.33,
    adaface: 0.33,
    elasticface: 0.34,
  });

  const [serviceStatus, setServiceStatus] = useState(null);
  const [isLoadingStatus, setIsLoadingStatus] = useState(false);

  const captureImage = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setCapturedImage(imageSrc);
  };

  const captureSecondImage = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setSecondImage(imageSrc);
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
      [model]: parseFloat(value),
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
      formData.append("image1", image1);
      formData.append("image2", image2);
      formData.append("model_weights", JSON.stringify(modelWeights));

      // ส่งคำขอไปยัง API
      const response = await axios.post(
        `${API_URL}/api/v1/face-recognition/compare`,
        formData
      );

      setResult(response.data);
    } catch (error) {
      console.error("Error comparing faces:", error);
      setResult({ error: "Failed to compare faces" });
    } finally {
      setIsLoading(false);
    }
  };

  const checkSecurity = async () => {
    if (!capturedImage) return;

    setIsLoading(true);

    try {
      // แปลงรูปภาพจาก Data URL เป็น Blob
      const fetchImage = async (dataUrl) => {
        const response = await fetch(dataUrl);
        const blob = await response.blob();
        return blob;
      };

      const image = await fetchImage(capturedImage);

      // สร้าง FormData
      const formData = new FormData();
      formData.append("image", image);
      formData.append("checks", "liveness,deepfake,spoofing");

      // ส่งคำขอไปยัง API
      const response = await axios.post(
        `${API_URL}/api/v1/security/check`,
        formData
      );

      setResult(response.data);
    } catch (error) {
      console.error("Error checking security:", error);
      setResult({ error: "Failed to check security" });
    } finally {
      setIsLoading(false);
    }
  };

  const detectFaces = async () => {
    if (!capturedImage) return;

    setIsLoading(true);

    try {
      // แปลงรูปภาพจาก Data URL เป็น Blob
      const fetchImage = async (dataUrl) => {
        const response = await fetch(dataUrl);
        const blob = await response.blob();
        return blob;
      };

      const image = await fetchImage(capturedImage);

      // สร้าง FormData
      const formData = new FormData();
      formData.append("image", image);

      // ส่งคำขอไปยัง API
      const response = await axios.post(
        `${API_URL}/api/v1/face-detection`,
        formData
      );

      setResult(response.data);
    } catch (error) {
      console.error("Error detecting faces:", error);
      setResult({ error: "Failed to detect faces" });
    } finally {
      setIsLoading(false);
    }
  };

  const checkServiceStatus = async () => {
    setIsLoadingStatus(true);
    try {
      const response = await axios.get(`${API_URL}/api/v1/status`);
      setServiceStatus(response.data);
    } catch (error) {
      console.error("Error checking service status:", error);
      setServiceStatus({
        timestamp: new Date().toISOString(),
        services: {
          "face-detection": {
            status: "offline",
            message: "ไม่สามารถเชื่อมต่อได้",
          },
          "face-recognition": {
            status: "offline",
            message: "ไม่สามารถเชื่อมต่อได้",
          },
          liveness: { status: "offline", message: "ไม่สามารถเชื่อมต่อได้" },
          deepfake: { status: "offline", message: "ไม่สามารถเชื่อมต่อได้" },
        },
      });
    } finally {
      setIsLoadingStatus(false);
    }
  };

  const getServiceThaiName = (service) => {
    const serviceNames = {
      "face-detection": "การตรวจจับใบหน้า",
      "face-recognition": "การจดจำใบหน้า",
      liveness: "การตรวจจับความมีชีวิต",
      deepfake: "การตรวจจับภาพปลอม Deepfake",
    };
    return serviceNames[service] || service;
  };

  useEffect(() => {
    if (activeTab === "status") {
      checkServiceStatus();
    }
  }, [activeTab]);

  const analyzeImage = () => {
    if (activeTab === "recognition") {
      compareFaces();
    } else if (activeTab === "security") {
      checkSecurity();
    } else if (activeTab === "detection") {
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
          <Row>
            <Col md={6}>
              <Card className="mb-3">
                <Card.Body>
                  <h5>First Image</h5>
                  {capturedImage ? (
                    <img
                      src={capturedImage}
                      alt="Captured"
                      className="img-fluid rounded"
                    />
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
                      <Button
                        variant="outline-secondary"
                        onClick={() => setCapturedImage(null)}
                      >
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
                    <img
                      src={secondImage}
                      alt="Second"
                      className="img-fluid rounded"
                    />
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
                      <Button
                        variant="outline-secondary"
                        onClick={() => setSecondImage(null)}
                      >
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
              {isLoading ? "Processing..." : "Compare Faces"}
            </Button>
            <Button variant="secondary" onClick={reset} disabled={isLoading}>
              Reset
            </Button>
          </div>

          {result && !result.error && (
            <Card className="result-container">
              <Card.Body>
                <h4>Comparison Result</h4>
                <Alert variant={result.is_match ? "success" : "danger"}>
                  {result.is_match
                    ? "The faces match!"
                    : "The faces do not match!"}
                </Alert>
                <p>
                  <strong>Similarity Score:</strong>{" "}
                  {(result.similarity * 100).toFixed(2)}%
                </p>

                <h5 className="mt-3">Model Details</h5>
                {result.model_details &&
                  Object.entries(result.model_details).map(([model, score]) => (
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
                      onChange={(e) =>
                        handleWeightChange(model, e.target.value)
                      }
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
          <Row>
            <Col md={6}>
              <Card className="mb-3">
                <Card.Body>
                  <h5>Capture Image</h5>
                  {capturedImage ? (
                    <img
                      src={capturedImage}
                      alt="Captured"
                      className="img-fluid rounded"
                    />
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
                      <Button
                        variant="outline-secondary"
                        onClick={() => setCapturedImage(null)}
                      >
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
                      {isLoading ? "Processing..." : "Run Security Checks"}
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
                <Alert variant={result.is_real_face ? "success" : "danger"}>
                  {result.is_real_face
                    ? "This appears to be a real face!"
                    : "This does NOT appear to be a real face!"}
                </Alert>

                {result.liveness && (
                  <div className="mb-3">
                    <h5>Liveness Check</h5>
                    <p>
                      <strong>Result:</strong>{" "}
                      {result.liveness.is_live
                        ? "Live Face"
                        : "Not a Live Face"}
                    </p>
                    <p>
                      <strong>Confidence:</strong>{" "}
                      {(result.liveness.score * 100).toFixed(2)}%
                    </p>
                  </div>
                )}

                {result.deepfake && (
                  <div className="mb-3">
                    <h5>Deepfake Check</h5>
                    <p>
                      <strong>Result:</strong>{" "}
                      {result.deepfake.is_fake
                        ? "Likely Deepfake"
                        : "Not a Deepfake"}
                    </p>
                    <p>
                      <strong>Confidence:</strong>{" "}
                      {(result.deepfake.score * 100).toFixed(2)}%
                    </p>
                  </div>
                )}

                {result.spoofing && (
                  <div className="mb-3">
                    <h5>Spoofing Check</h5>
                    <p>
                      <strong>Result:</strong>{" "}
                      {result.spoofing.is_attack
                        ? "Likely Spoofing Attack"
                        : "Not a Spoofing Attack"}
                    </p>
                    <p>
                      <strong>Confidence:</strong>{" "}
                      {(result.spoofing.score * 100).toFixed(2)}%
                    </p>
                  </div>
                )}
              </Card.Body>
            </Card>
          )}
        </Tab>

        <Tab eventKey="detection" title="Face Detection">
          <Row>
            <Col md={12}>
              <Card className="mb-3">
                <Card.Body>
                  <h5>Capture Image</h5>
                  {capturedImage ? (
                    <img
                      src={capturedImage}
                      alt="Captured"
                      className="img-fluid rounded"
                    />
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
                      <Button
                        variant="outline-secondary"
                        onClick={() => setCapturedImage(null)}
                      >
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
                        {isLoading ? "Processing..." : "Detect Faces"}
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
                        alt={`Face ${index + 1}`}
                        className="face-image"
                      />
                      <p className="mt-2 mb-0">Face #{index + 1}</p>
                      <small>
                        Confidence: {(face.confidence * 100).toFixed(1)}%
                      </small>
                    </div>
                  ))}
                </div>
              </Card.Body>
            </Card>
          )}
        </Tab>

        <Tab eventKey="status" title="สถานะโมเดล">
          <Card className="mb-3">
            <Card.Body>
              <h5>สถานะของโมเดลและบริการ</h5>
              <p>สถานะปัจจุบันของโมเดลและบริการทั้งหมดในระบบ</p>

              <Button
                variant="primary"
                onClick={checkServiceStatus}
                disabled={isLoadingStatus}
              >
                {isLoadingStatus ? "กำลังโหลด..." : "รีเฟรชสถานะ"}
              </Button>

              {serviceStatus && (
                <div className="mt-4">
                  <Table striped bordered hover>
                    <thead>
                      <tr>
                        <th>บริการ</th>
                        <th>สถานะ</th>
                        <th>โมเดลที่โหลด</th>
                        <th>เวอร์ชัน</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(serviceStatus.services).map(
                        ([service, details]) => (
                          <tr key={service}>
                            <td>{getServiceThaiName(service)}</td>
                            <td>
                              <Badge
                                bg={
                                  details.status === "online"
                                    ? "success"
                                    : "danger"
                                }
                              >
                                {details.status === "online"
                                  ? "พร้อมใช้งาน"
                                  : "ไม่พร้อมใช้งาน"}
                              </Badge>
                            </td>
                            <td>
                              {details.models && details.models.length > 0 ? (
                                <ul className="mb-0 ps-3">
                                  {details.models.map((model) => (
                                    <li key={model}>{model}</li>
                                  ))}
                                </ul>
                              ) : (
                                <span className="text-muted">-</span>
                              )}
                            </td>
                            <td>{details.version || "-"}</td>
                          </tr>
                        )
                      )}
                    </tbody>
                  </Table>
                  <small className="text-muted">
                    อัพเดทล่าสุด:{" "}
                    {new Date(serviceStatus.timestamp).toLocaleString("th-TH")}
                  </small>
                </div>
              )}
            </Card.Body>
          </Card>
        </Tab>

        <Tab eventKey="realtime" title="วิเคราะห์แบบเรียลไทม์">
          <RealtimeFaceAnalysis />
        </Tab>
      </Tabs>

      {result && result.error && (
        <Alert variant="danger">Error: {result.error}</Alert>
      )}
    </Container>
  );
}

export default App;
