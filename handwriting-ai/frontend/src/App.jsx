import axios from 'axios'
import { motion } from 'framer-motion'
import { useRef, useState } from 'react'

export default function App() {
  const canvasRef = useRef(null)
  const [drawing, setDrawing] = useState(false)
  const [result, setResult] = useState('')
  const [confidence, setConfidence] = useState(0)

  const startDraw = (e) => {
    const ctx = canvasRef.current.getContext('2d')
    ctx.beginPath()
    ctx.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY)
    setDrawing(true)
  }

  const draw = (e) => {
    if (!drawing) return

    const ctx = canvasRef.current.getContext('2d')

    ctx.lineWidth = 14
    ctx.lineCap = 'round'
    ctx.strokeStyle = 'white'

    ctx.lineTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY)
    ctx.stroke()
  }

  const stopDraw = () => {
    setDrawing(false)
  }

  const clearCanvas = () => {
    const ctx = canvasRef.current.getContext('2d')
    ctx.clearRect(0, 0, 400, 400)
  }

  const predict = async () => {
    try {
      canvasRef.current.toBlob(async (blob) => {
        try {
          const formData = new FormData()
          formData.append('image', blob)

          const res = await axios.post(
            'http://127.0.0.1:5000/predict',
            formData,
            {
              headers: {
                'Content-Type': 'multipart/form-data',
              },
            }
          )

          setResult(res.data.letter)
          setConfidence((res.data.confidence * 100).toFixed(2))
        } catch (error) {
          console.error('Prediction error:', error)
          alert('Lỗi: ' + (error.response?.data?.error || error.message || 'Không thể dự đoán'))
        }
      })
    } catch (error) {
      console.error('Canvas error:', error)
      alert('Lỗi canvas: ' + error.message)
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8">
      <motion.h1
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-5xl font-bold mb-10"
      >
        AI Handwriting Recognition
      </motion.h1>

      <motion.canvas
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5 }}
        ref={canvasRef}
        width={400}
        height={400}
        className="border-4 border-cyan-400 rounded-3xl bg-black shadow-2xl"
        onMouseDown={startDraw}
        onMouseMove={draw}
        onMouseUp={stopDraw}
        onMouseLeave={stopDraw}
      />

      <div className="flex gap-4 mt-6">
        <button
          onClick={predict}
          className="px-8 py-3 bg-cyan-500 rounded-2xl text-lg hover:scale-105 transition"
        >
          Predict
        </button>

        <button
          onClick={clearCanvas}
          className="px-8 py-3 bg-red-500 rounded-2xl text-lg hover:scale-105 transition"
        >
          Clear
        </button>
      </div>

      {result && (
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-10 p-8 rounded-3xl bg-white/10 backdrop-blur-lg shadow-2xl text-center"
        >
          <h2 className="text-2xl">Prediction</h2>

          <p className="text-8xl font-bold text-cyan-400 mt-4">
            {result}
          </p>

          <p className="mt-4 text-xl">
            Confidence: {confidence}%
          </p>
        </motion.div>
      )}
    </div>
  )
}
