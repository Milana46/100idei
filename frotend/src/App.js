import React, { useState, useMemo } from "react";
import "./App.css";


export default function App() {
  const [diagnosis, setDiagnosis] = useState("");
  const [microorganism, setMicroorganism] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [data, setData] = useState(null);

  const bugs = useMemo(() => {
    const count = 55; // можно 40..90
    return Array.from({ length: count }).map((_, i) => {
      const size = 8 + Math.random() * 22; // 8..30px
      const left = Math.random() * 100; // %
      const top = Math.random() * 100; // %
      const dur = 10 + Math.random() * 22; // 10..32s
      const delay = -Math.random() * dur; // чтобы не стартовали синхронно
      const drift = 10 + Math.random() * 40; // амплитуда
      const rot = Math.random() * 360;

      // чуть разный “тип”
      const type = Math.random() < 0.6 ? "coccus" : "bacillus";

      return { id: i, size, left, top, dur, delay, drift, rot, type };
    });
  }, []);


  const API_URL = "http://localhost:5000/predict";

  // Обработчик отправки данных
  async function handleSubmit(e) {
    e.preventDefault();
    setErr("");
    setData(null);
    setLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ diagnosis, microorganism }),
      });

      const json = await response.json();

      if (!response.ok) {
        throw new Error(json.error || "Request failed");
      }

      setData(json);
    } catch (e2) {
      setErr(e2.message);
    } finally {
      setLoading(false);
    }

  }

  
  return (
    <div className="page">

      <div className="bgMany" aria-hidden="true">
  {bugs.map((b) => (
    <span
      key={b.id}
      className={`bug ${b.type}`}
      style={{
        left: `${b.left}%`,
        top: `${b.top}%`,
        width: `${b.size}px`,
        height: `${b.size}px`,
        animationDuration: `${b.dur}s`,
        animationDelay: `${b.delay}s`,
        "--drift": `${b.drift}px`,
        "--rot": `${b.rot}deg`,
      }}
    />
  ))}
  <div className="grain" />
</div>
    <div className="bg">
      <div className="blob b1"></div>
      <div className="blob b2"></div>
      <div className="blob b3"></div>
      <div className="blob b4"></div>
      <div className="blob b5"></div>
      <div className="grain"></div>
    </div>
      
      <div className="card">
        <h1 className="title">Антибиотик-рекомендатор</h1>
        <p className="subtitle">
          Введите диагноз и микроорганизм — система предложит TOP-15 антибиотиков
          с оценкой вероятностей (S/I/R).
        </p>

        {/* Форма для ввода данных */}
        <form className="form" onSubmit={handleSubmit}>
          <label className="label">
            Диагноз
            <input
              className="input"
              value={diagnosis}
              onChange={(e) => setDiagnosis(e.target.value)}
              placeholder="Напр.: абсцесс"
              required
            />
          </label>

          <label className="label">
            Микроорганизм
            <input
              className="input"
              value={microorganism}
              onChange={(e) => setMicroorganism(e.target.value)}
              placeholder="Напр.: kpn"
              required
            />
          </label>

          <button className="btn" disabled={loading}>
            {loading ? "Рассчитываем..." : "Получить рекомендации"}
          </button>

          {err && <div className="error">Ошибка: {err}</div>}
        </form>
      </div>

      {/* Вывод рекомендаций */}
      {data && (
        <div className="card">
          <h2 className="title2">Результаты</h2>
          <div className="meta">
            <div><b>Диагноз:</b> {data.diagnosis}</div>
            <div><b>Микроорганизм:</b> {data.microorganism}</div>
          </div>

          <table className="table">
            <thead>
              <tr>
                <th>#</th>
                <th>Антибиотик</th>
                <th>P(S)</th>
                <th>P(I)</th>
                <th>P(R)</th>
              </tr>
            </thead>
            <tbody>
              {data.top15.map((row, idx) => (
                <tr key={idx}>
                  <td>{idx + 1}</td>
                  <td>{row["Антибиотик"]}</td>
                  <td>{row["P(S)"]}</td>
                  <td>{row["P(I)"]}</td>
                  <td>{row["P(R)"]}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <p className="hint">
            Примечание: это инструмент поддержки решений и не заменяет врача.
          </p>
        </div>
      )}
    </div>
  );
}
