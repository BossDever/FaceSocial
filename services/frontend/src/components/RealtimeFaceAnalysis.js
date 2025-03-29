import React, { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";

const API_URL = "http://localhost:8000";

const RealtimeFaceAnalysis = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [isActive, setIsActive] = useState(false);
  const [analysisResults, setAnalysisResults] = useState({});
  const [processFrames, setProcessFrames] = useState(true);
  const [fps, setFps] = useState(0);
  const [status, setStatus] = useState("กำลังเริ่มต้น...");

  // เก็บเวลาที่ใช้ในการประมวลผลแต่ละเฟรม
  const [processingTimes, setProcessingTimes] = useState([]);

  // จำกัดระยะเวลาการส่ง API requests (ส่งทุกๆ 300ms)
  const API_THROTTLE = 300;
  const lastApiCallRef = useRef(0);

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user",
  };

  // ฟังก์ชันวาดกรอบและข้อมูลบนใบหน้า
  const drawResults = (ctx, results) => {
    // ล้างแคนวาส
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    if (!results || !results.faces || results.faces.length === 0) return;

    results.faces.forEach((face, index) => {
      const [x, y, width, height] = face.bbox;

      // วาดกรอบรอบใบหน้า
      ctx.strokeStyle = "#00FF00";
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      // ข้อมูลพื้นฐาน
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.fillRect(x + width + 5, y, 170, 120);

      ctx.font = "14px Arial";
      ctx.fillStyle = "#FFFFFF";

      // แสดงเพศและอายุ (ถ้ามี)
      if (face.gender) {
        ctx.fillText(
          `เพศ: ${face.gender === "male" ? "ชาย" : "หญิง"}`,
          x + width + 10,
          y + 20
        );
      }

      if (face.age) {
        ctx.fillText(`อายุ: ${face.age} ปี`, x + width + 10, y + 40);
      }

      // แสดงผลการตรวจสอบความปลอดภัย (จากข้อมูล analysisResults)
      if (analysisResults.liveness) {
        const liveStatus = analysisResults.liveness.is_live
          ? "มีชีวิต ✓"
          : "ไม่มีชีวิต ✗";
        const liveColor = analysisResults.liveness.is_live
          ? "#00FF00"
          : "#FF0000";
        ctx.fillStyle = liveColor;
        ctx.fillText(`สถานะ: ${liveStatus}`, x + width + 10, y + 60);
        ctx.fillStyle = "#FFFFFF";
      }

      if (analysisResults.deepfake) {
        const deepfakeStatus = analysisResults.deepfake.is_fake
          ? "ภาพปลอม ✗"
          : "ภาพจริง ✓";
        const deepfakeColor = !analysisResults.deepfake.is_fake
          ? "#00FF00"
          : "#FF0000";
        ctx.fillStyle = deepfakeColor;
        ctx.fillText(`Deepfake: ${deepfakeStatus}`, x + width + 10, y + 80);
        ctx.fillStyle = "#FFFFFF";
      }

      ctx.fillText(
        `ความเชื่อมั่น: ${(face.confidence * 100).toFixed(0)}%`,
        x + width + 10,
        y + 100
      );
    });
  };

  // ตรวจจับใบหน้าจากเฟรมปัจจุบัน
  const detectFaces = async (frameData) => {
    try {
      const now = Date.now();
      if (now - lastApiCallRef.current < API_THROTTLE) return;

      lastApiCallRef.current = now;
      setStatus("กำลังวิเคราะห์...");

      const startTime = performance.now();

      // ส่งเฟรมไปตรวจจับใบหน้า
      const detectResponse = await axios.post(
        `${API_URL}/api/v1/face-detection`,
        {
          image: frameData.split(",")[1],
          include_attributes: true,
        }
      );

      if (
        !detectResponse.data ||
        !detectResponse.data.faces ||
        detectResponse.data.faces.length === 0
      ) {
        setStatus("ไม่พบใบหน้า");
        return;
      }

      // ถ้ามีใบหน้า ทำการตรวจสอบความปลอดภัย
      const securityResponse = await axios.post(
        `${API_URL}/api/v1/security/check`,
        {
          image: frameData.split(",")[1],
          checks: "liveness,deepfake,spoofing",
        }
      );

      // รวมผลลัพธ์ทั้งหมด
      const result = {
        ...detectResponse.data,
        ...securityResponse.data,
      };

      const endTime = performance.now();
      const processingTime = endTime - startTime;

      // เก็บเวลาในการประมวลผลและคำนวณ FPS ใหม่
      setProcessingTimes((prev) => {
        const newTimes = [...prev, processingTime].slice(-10);
        const avgTime =
          newTimes.reduce((sum, time) => sum + time, 0) / newTimes.length;
        setFps(Math.round(1000 / avgTime));
        return newTimes;
      });

      setAnalysisResults(result);
      setStatus("พร้อมใช้งาน");

      return result;
    } catch (error) {
      console.error("Error analyzing frame:", error);
      setStatus("เกิดข้อผิดพลาด");
      return null;
    }
  };

  // ล้างแคนวาสเมื่อเริ่มต้นหรือหยุดการวิเคราะห์
  useEffect(() => {
    if (!isActive && canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    }
  }, [isActive]);

  // ทำการประมวลผลแต่ละเฟรม
  useEffect(() => {
    let requestId;
    let lastFrameTime = 0;

    const processFrame = async () => {
      if (
        !webcamRef.current ||
        !canvasRef.current ||
        !isActive ||
        !processFrames
      ) {
        requestId = requestAnimationFrame(processFrame);
        return;
      }

      const now = performance.now();
      const frameDelta = now - lastFrameTime;

      // จำกัดการประมวลผลไม่ให้เกิน 10 FPS (ทุก 100ms)
      if (frameDelta >= 100) {
        lastFrameTime = now;
        const imageData = webcamRef.current.getScreenshot();

        if (imageData) {
          const results = await detectFaces(imageData);

          if (results && canvasRef.current) {
            const ctx = canvasRef.current.getContext("2d");
            drawResults(ctx, results);
          }
        }
      }

      requestId = requestAnimationFrame(processFrame);
    };

    if (isActive) {
      requestId = requestAnimationFrame(processFrame);
    }

    return () => {
      cancelAnimationFrame(requestId);
    };
  }, [isActive, processFrames]);

  // ตั้งค่าขนาดแคนวาสเมื่อคอมโพเนนต์ถูกโหลด
  useEffect(() => {
    if (canvasRef.current && webcamRef.current) {
      const canvas = canvasRef.current;
      canvas.width = 640;
      canvas.height = 480;
    }
  }, []);

  return (
    <div className="realtime-analysis">
      <div className="d-flex justify-content-between mb-2">
        <h3>การวิเคราะห์ใบหน้าแบบเรียลไทม์</h3>
        <div>
          <button
            className={`btn ${isActive ? "btn-danger" : "btn-success"} me-2`}
            onClick={() => setIsActive(!isActive)}
          >
            {isActive ? "หยุด" : "เริ่ม"} การวิเคราะห์
          </button>
        </div>
      </div>

      <div className="video-container position-relative">
        <Webcam
          audio={false}
          ref={webcamRef}
          videoConstraints={videoConstraints}
          className="webcam"
          width={640}
          height={480}
        />
        <canvas
          ref={canvasRef}
          className="position-absolute top-0 left-0"
          style={{ opacity: isActive ? 1 : 0 }}
        />

        <div className="status-overlay position-absolute bottom-0 left-0 p-2 bg-dark text-white w-100">
          <div className="d-flex justify-content-between">
            <span>สถานะ: {status}</span>
            <span>FPS: {fps}</span>
            <span>ใบหน้าที่พบ: {analysisResults.faces?.length || 0}</span>
          </div>
        </div>
      </div>

      <div className="mt-3">
        <div className="form-check">
          <input
            className="form-check-input"
            type="checkbox"
            id="processFramesToggle"
            checked={processFrames}
            onChange={(e) => setProcessFrames(e.target.checked)}
          />
          <label className="form-check-label" htmlFor="processFramesToggle">
            ประมวลผลวิดีโอแบบเรียลไทม์
          </label>
        </div>
      </div>

      <style jsx>{`
        .realtime-analysis {
          margin-top: 1rem;
        }

        .video-container {
          width: 640px;
          height: 480px;
          margin: 0 auto;
          overflow: hidden;
          border: 1px solid #ddd;
          border-radius: 4px;
        }

        .webcam {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }

        .status-overlay {
          opacity: 0.8;
        }
      `}</style>
    </div>
  );
};

export default RealtimeFaceAnalysis;
