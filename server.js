const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const { Canvas, Image, ImageData } = require('canvas');
const faceapi = require('face-api.js');
const tf = require('@tensorflow/tfjs');

const app = express();
const port = 3000;

// Habilitar CORS para todas las peticiones
app.use(cors());

// Configuración de multer (recibe imágenes temporales)
const upload = multer({ storage: multer.memoryStorage() });

// Necesario para que face-api funcione en Node
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

let faceMatcher; // se carga una vez

// Cargar modelos y descriptores conocidos
async function loadModelsAndDescriptors() {
  const MODEL_PATH = path.join(__dirname, 'models');

  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH);

  const labeledImages = [
    { name: 'Silvio', file: 'silvio.png' },
    { name: 'Jessica', file: 'jessica.jpg' },
    { name: 'Chavez', file: 'chavez.jpg' }
  ];

  const labeledDescriptors = [];

  for (const { name, file } of labeledImages) {
    const imgPath = path.join(__dirname, 'labeled_images', file);
    const img = await canvasLoadImage(imgPath);

    const detection = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection) continue;

    labeledDescriptors.push(
      new faceapi.LabeledFaceDescriptors(name, [detection.descriptor])
    );
  }

  faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
}

async function canvasLoadImage(filePath) {
  const buffer = fs.readFileSync(filePath);
  const img = new Image();
  img.src = buffer;
  return img;
}

// Ruta para recibir imagen desde el frontend
app.post('/api/recognize', upload.single('image'), async (req, res) => {
  try {
    const buffer = req.file.buffer;
    const img = new Image();
    img.src = buffer;

    const detection = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection) {
      return res.json({ success: false, message: 'No se detectó rostro' });
    }

    const bestMatch = faceMatcher.findBestMatch(detection.descriptor);

    return res.json({
      success: true,
      person: bestMatch.label,
      distance: bestMatch.distance
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, error: 'Error interno del servidor' });
  }
});

// Iniciar servidor
loadModelsAndDescriptors().then(() => {
  app.listen(port, () => {
    console.log(`Servidor iniciado en http://localhost:${port}`);
  });
});