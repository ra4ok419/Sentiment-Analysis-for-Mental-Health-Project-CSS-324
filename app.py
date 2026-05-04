import gradio as gr
from transformers import pipeline

LABEL_MAP = {
    "LABEL_0": "Anxiety",
    "LABEL_1": "Bipolar",
    "LABEL_2": "Depression",
    "LABEL_3": "Normal",
    "LABEL_4": "Personality Disorder",
    "LABEL_5": "Stress",
    "LABEL_6": "Suicidal",
}

classifier = pipeline(
    "text-classification",
    model="mental_bert_final",
    top_k=None
)

def predict(text):
    if not text.strip():
        return "Введите текст для анализа"

    results = sorted(classifier(text)[0], key=lambda x: x["score"], reverse=True)
    label = LABEL_MAP.get(results[0]["label"], results[0]["label"])
    score = results[0]["score"]

    if label in ["Suicidal", "Depression"]:
        risk_html = f"""
        <div style='background:#2d1b1b;border-left:4px solid #ff4444;padding:16px;border-radius:8px;margin-bottom:12px'>
            <span style='font-size:24px'>🔴</span>
            <span style='color:#ff6b6b;font-size:20px;font-weight:bold;margin-left:8px'>{label}</span>
            <span style='color:#aaa;margin-left:8px'>· Высокий риск · {score:.0%}</span>
        </div>
        <div style='background:#1e1e1e;padding:12px;border-radius:8px;color:#ff9999;font-size:14px'>
            ⚠️ Если вам тяжело — позвоните: <b>8-800-2000-122</b> (бесплатно, круглосуточно)
        </div>
        """
    elif label == "Normal":
        risk_html = f"""
        <div style='background:#1b2d1b;border-left:4px solid #44ff88;padding:16px;border-radius:8px'>
            <span style='font-size:24px'>🟢</span>
            <span style='color:#6bff9e;font-size:20px;font-weight:bold;margin-left:8px'>{label}</span>
            <span style='color:#aaa;margin-left:8px'>· Всё хорошо · {score:.0%}</span>
        </div>
        """
    else:
        risk_html = f"""
        <div style='background:#2d2a1b;border-left:4px solid #ffaa44;padding:16px;border-radius:8px'>
            <span style='font-size:24px'>🟡</span>
            <span style='color:#ffc97a;font-size:20px;font-weight:bold;margin-left:8px'>{label}</span>
            <span style='color:#aaa;margin-left:8px'>· Средний риск · {score:.0%}</span>
        </div>
        """

    # Прогресс-бары для всех классов
    bars_html = "<div style='margin-top:16px'>"
    for r in results:
        name = LABEL_MAP.get(r["label"], r["label"])
        pct  = r["score"] * 100
        color = "#ff4444" if name in ["Suicidal","Depression"] else \
                "#44ff88" if name == "Normal" else "#ffaa44"
        bars_html += f"""
        <div style='margin-bottom:8px'>
            <div style='display:flex;justify-content:space-between;color:#ccc;font-size:13px;margin-bottom:3px'>
                <span>{name}</span><span>{pct:.1f}%</span>
            </div>
            <div style='background:#333;border-radius:4px;height:6px'>
                <div style='background:{color};width:{pct}%;height:6px;border-radius:4px;transition:width 0.3s'></div>
            </div>
        </div>
        """
    bars_html += "</div>"

    return risk_html + bars_html

css = """
.gradio-container { max-width: 800px !important; margin: auto !important; }
footer { display: none !important; }
"""

with gr.Blocks() as demo:
    gr.HTML("""
        <div style='text-align:center;padding:24px 0 8px'>
            <div style='font-size:40px'>🧠</div>
            <h1 style='color:#fff;font-size:28px;margin:8px 0'>Mental Health Classifier</h1>
            <p style='color:#888;font-size:15px'>Напишите как вы себя чувствуете — модель определит ваше состояние</p>
        </div>
    """)

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                lines=6,
                placeholder="Напишите здесь что угодно — как прошёл день, что вас беспокоит, любые мысли...",
                label="Ваш текст",
                show_label=False
            )
            btn = gr.Button("Анализировать →", variant="primary", size="lg")

        with gr.Column():
            out = gr.HTML(label="Результат")

    btn.click(fn=predict, inputs=text, outputs=out)
    text.submit(fn=predict, inputs=text, outputs=out)

    gr.HTML("""
        <div style='text-align:center;color:#555;font-size:12px;padding:16px'>
            MARS Team · CSS 324 · Модель: Fine-tuned MentalBERT
        </div>
    """)

    demo.launch(css=css, theme=gr.themes.Base(), share=True)