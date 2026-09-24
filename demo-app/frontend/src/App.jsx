import { useEffect, useMemo, useState } from 'react';
import {
  Activity, AlertCircle, ArrowDownToLine, Check, CheckCircle2, ChevronDown,
  CircleHelp, Cpu, Gauge, LoaderCircle, Play, RotateCcw, ShieldCheck,
  SquareTerminal, Zap,
} from 'lucide-react';
import scenarios from './scenarios.json';

const API = import.meta.env.VITE_API_BASE || '';
const apiUrl = (path) => `${API}${path}`;

function optionLabel(question, key) {
  if (question.type === 'score') return question.criteria?.[Number(key)] ?? key;
  if (question.type === 'choice') return question.criteria?.[key] ?? key;
  return key === 'true' ? '是' : '否';
}

function predictedLabel(answer) {
  if (!answer) return null;
  if (answer.type === 'choice') return String(answer.choice ?? answer.label ?? '');
  if (answer.type === 'score') {
    const entries = Object.entries(answer.probabilities || {});
    return entries.sort((a, b) => b[1] - a[1])[0]?.[0] ?? String(Math.round(answer.score ?? 0));
  }
  return Number(answer.noul ?? 0) >= 0.5 ? 'true' : 'false';
}

function expectedLabel(reference, questionId) {
  const value = reference?.[questionId];
  if (value === undefined || value === null) return null;
  return typeof value === 'boolean' ? (value ? 'true' : 'false') : String(value);
}

function percent(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(1)}%`;
}

function resultRows(data, scenario, reference) {
  return Object.entries(scenario.questions).map(([id, question]) => {
    const answer = data?.answers?.[id];
    const predicted = predictedLabel(answer);
    const expected = expectedLabel(reference, id);
    return { id, question, answer, predicted, expected, correct: expected === null ? null : predicted === expected };
  });
}

function ReferencePill({ value, question }) {
  if (value === null) return <span className="ref muted">自定义样例</span>;
  return <span className="ref"><span>参考</span>{optionLabel(question, value)}</span>;
}

function AnswerCard({ row }) {
  const { id, question, answer, predicted, expected, correct } = row;
  if (!answer) {
    return <article className="answer-card">
      <div className="answer-top"><div><span className="qid">{id}</span><h3>{question.instructions}</h3></div><span className="type-tag">{question.type}</span></div>
      <div className="answer-empty">运行模型后，这个问题的类型化答案和选项概率会显示在这里。</div>
    </article>;
  }

  const options = question.type === 'noul'
    ? [['false', answer.probabilities?.false ?? 1 - (answer.noul ?? 0)], ['true', answer.probabilities?.true ?? answer.noul ?? 0]]
    : Object.entries(answer.probabilities || {});
  const topProbability = question.type === 'noul'
    ? Math.max(answer.noul ?? 0, 1 - (answer.noul ?? 0))
    : answer.probabilities?.[predicted] ?? 0;
  const displayed = question.type === 'score'
    ? `${optionLabel(question, predicted)} · E[score] ${(answer.score ?? 0).toFixed(2)}`
    : question.type === 'noul' ? ((answer.noul ?? 0) >= 0.5 ? '是' : '否') : optionLabel(question, predicted);

  return <article className="answer-card">
    <div className="answer-top">
      <div><span className="qid">{id}</span><h3>{question.instructions}</h3></div>
      <span className="type-tag">{question.type}</span>
    </div>
    <div className="answer-main">
      <div className="chosen-answer"><span>模型答案</span><strong>{displayed}</strong></div>
      <div className="answer-badges">
        <span className="prob-badge"><Zap size={13} /> P(所选项) {percent(topProbability)}</span>
        <span className="confidence-badge">answer_confidence {answer.answer_confidence == null ? '—' : percent(answer.answer_confidence)}</span>
        <span className="confidence-badge subtle">confidence {answer.confidence == null ? '—' : percent(answer.confidence)}</span>
        {correct === null ? <span className="ref muted">未计分</span> : correct
          ? <span className="result-mark good"><Check size={13} /> 一致</span>
          : <span className="result-mark bad"><AlertCircle size={13} /> 不一致</span>}
      </div>
    </div>
    <ReferencePill value={expected} question={question} />
    <div className="probability-list">
      {options.map(([key, value]) => {
        const chosen = String(key) === predicted;
        return <div className={`probability-row ${chosen ? 'chosen' : ''}`} key={key}>
          <span className="option-name" title={optionLabel(question, key)}>{optionLabel(question, key)}</span>
          <span className="prob-track"><i style={{ width: `${Math.max(0, Math.min(100, Number(value) * 100))}%` }} /></span>
          <span className="option-percent">{percent(Number(value))}</span>
        </div>;
      })}
    </div>
  </article>;
}

export default function App() {
  const [scenarioIndex, setScenarioIndex] = useState(0);
  const [sampleIndex, setSampleIndex] = useState(0);
  const [stateText, setStateText] = useState(scenarios[0].samples[0].text);
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState('');
  const [result, setResult] = useState(null);
  const [records, setRecords] = useState([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState('');
  const [error, setError] = useState('');
  const [showRaw, setShowRaw] = useState(false);
  const [threshold, setThreshold] = useState(0.7);

  const scenario = scenarios[scenarioIndex];
  const sample = scenario.samples[sampleIndex];
  const edited = stateText !== sample.text;

  async function checkHealth() {
    setHealthError('');
    try {
      const response = await fetch(apiUrl('/api/health'));
      const body = await response.json();
      if (!response.ok || body.status !== 'ok') throw new Error(body.detail || `HTTP ${response.status}`);
      setHealth(body);
    } catch (err) {
      setHealth(null);
      setHealthError(err.message || '无法连接 API');
    }
  }

  useEffect(() => { checkHealth(); }, []);

  function chooseScenario(index) {
    setScenarioIndex(index);
    setSampleIndex(0);
    setStateText(scenarios[index].samples[0].text);
    setResult(null);
    setError('');
    setShowRaw(false);
  }

  function chooseSample(index) {
    setSampleIndex(index);
    setStateText(scenario.samples[index].text);
    setResult(null);
    setError('');
    setShowRaw(false);
  }

  async function callModel(targetScenario, targetSample, text) {
    const response = await fetch(apiUrl('/api/decisions'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        scenario_id: targetScenario.id,
        model: health?.model || 'laya-typed-decisions-zh',
        state: { message: text, context: targetScenario.context },
        questions: targetScenario.questions,
      }),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail || body);
      throw new Error(`HTTP ${response.status} · ${detail || response.statusText}`);
    }
    if (!body.answers) throw new Error('API 返回中没有 answers 字段');
    return { data: body, latency: Number(body.latency_ms ?? 0), sampleName: targetSample.name };
  }

  function addRecord(targetScenario, targetSample, data, latency, reference) {
    const rows = resultRows(data, targetScenario, reference);
    const scored = rows.filter((row) => row.correct !== null);
    setRecords((current) => [{
      key: `${Date.now()}-${Math.random()}`,
      scenario: targetScenario.title,
      sample: targetSample.name,
      correct: scored.filter((row) => row.correct).length,
      total: scored.length,
      latency,
    }, ...current]);
  }

  async function runOne() {
    if (busy) return;
    setBusy(true); setError(''); setResult(null); setProgress('正在调用本机模型…');
    try {
      const prediction = await callModel(scenario, sample, stateText);
      const reference = edited ? null : sample.reference;
      setResult({ ...prediction, reference, scenarioId: scenario.id, sampleIndex });
      if (!edited) addRecord(scenario, sample, prediction.data, prediction.latency, reference);
      setHealth((current) => current ? { ...current, status: 'ok' } : current);
      setProgress('本条推理完成');
    } catch (err) {
      setError(err.message || '模型推理失败'); setProgress('请求失败');
    } finally { setBusy(false); }
  }

  async function runScenario() {
    if (busy) return;
    setBusy(true); setError(''); setResult(null);
    try {
      for (let i = 0; i < scenario.samples.length; i++) {
        const current = scenario.samples[i];
        setSampleIndex(i); setStateText(current.text); setProgress(`${scenario.title} · ${current.name} · ${i + 1}/${scenario.samples.length}`);
        const prediction = await callModel(scenario, current, current.text);
        setResult({ ...prediction, reference: current.reference, scenarioId: scenario.id, sampleIndex: i });
        addRecord(scenario, current, prediction.data, prediction.latency, current.reference);
      }
      setProgress(`${scenario.title} 场景完成`);
    } catch (err) { setError(err.message || '评测中断'); setProgress('评测中断'); }
    finally { setBusy(false); }
  }

  async function runAll() {
    if (busy) return;
    setBusy(true); setError(''); setRecords([]); setResult(null);
    let completed = 0;
    const total = scenarios.reduce((sum, item) => sum + item.samples.length, 0);
    try {
      for (let s = 0; s < scenarios.length; s++) {
        const targetScenario = scenarios[s];
        setScenarioIndex(s);
        for (let i = 0; i < targetScenario.samples.length; i++) {
          const targetSample = targetScenario.samples[i];
          setSampleIndex(i); setStateText(targetSample.text);
          setProgress(`全量评测 · ${targetScenario.title} / ${targetSample.name} · ${completed + 1}/${total}`);
          const prediction = await callModel(targetScenario, targetSample, targetSample.text);
          setResult({ ...prediction, reference: targetSample.reference, scenarioId: targetScenario.id, sampleIndex: i });
          addRecord(targetScenario, targetSample, prediction.data, prediction.latency, targetSample.reference);
          completed += 1;
        }
      }
      setProgress(`全部 ${total} 条样例已完成`);
    } catch (err) { setError(err.message || '全量评测中断'); setProgress(`已完成 ${completed}/${total} 条`); }
    finally { setBusy(false); }
  }

  const currentResult = result?.scenarioId === scenario.id && result?.sampleIndex === sampleIndex ? result : null;
  const answers = currentResult ? currentResult.data : null;
  const rows = useMemo(() => resultRows(answers, scenario, currentResult?.reference), [answers, scenario, currentResult]);
  const totalScored = records.reduce((sum, item) => sum + item.total, 0);
  const totalCorrect = records.reduce((sum, item) => sum + item.correct, 0);
  const averageLatency = records.length ? records.reduce((sum, item) => sum + item.latency, 0) / records.length : null;
  const actionAnswer = currentResult?.data?.answers?.[scenario.primary];
  const actionConfidence = actionAnswer?.type === 'noul'
    ? Math.max(actionAnswer.noul || 0, 1 - (actionAnswer.noul || 0))
    : actionAnswer?.probabilities?.[predictedLabel(actionAnswer)];

  return <main className="page-shell">
    <header className="topbar">
      <div className="brand"><span className="brand-mark">L</span><div><strong>Laya · 中文场景验证台</strong><small>TYPED DECISIONS · LOCAL INFERENCE</small></div></div>
      <div className="topbar-right"><span className={`service-pill ${health ? 'is-online' : 'is-offline'}`}><i />{health ? '本机模型已就绪' : '等待模型连接'}</span><button className="icon-button" onClick={checkHealth} title="检查接口" aria-label="检查接口"><RotateCcw size={15} /></button></div>
    </header>

    <section className="hero">
      <div><div className="eyebrow"><span /> 实验工作区</div><h1>验证模型怎么做判断。</h1><p>切换生活、Agent 与游戏场景，让同一个中文 Laya checkpoint 回答结构化问题；模型输出、概率与人工参考标签分开展示。</p></div>
      <div className="hero-stats">
        <div className="stat-chip"><span>CHECKPOINT</span><b>{health?.model || 'laya-typed-decisions-zh'}</b></div>
        <div className="stat-chip"><span>运行设备</span><b><Cpu size={14} /> {health?.gpu || health?.device || '检测中'}</b></div>
      </div>
    </section>

    <section className="decision-legend" aria-label="问题类型">
      <div><b className="mono">choice</b><span>候选项选择 <em>答案 + 每个选项的概率</em></span></div>
      <div><b className="mono">score</b><span>有序标准评分 <em>等级分布 + 期望分</em></span></div>
      <div><b className="mono">noul</b><span>二元判断 <em>P(true)</em></span></div>
    </section>

    <section className="connection-row">
      <div className={`connection-state ${health ? 'online' : 'offline'}`}><span className="status-dot" /><div><b>{health ? 'FastAPI 已连接' : 'FastAPI 未连接'}</b><small>{health ? `${health.device}${health.gpu ? ` · ${health.gpu}` : ''} · ${health.checkpoint}` : healthError || '请确认容器已启动且 checkpoint 已挂载'}</small></div></div>
      <button className="text-run" onClick={runAll} disabled={busy || !health}><Play size={14} fill="currentColor" /> 运行全部 {scenarios.reduce((sum, item) => sum + item.samples.length, 0)} 条</button>
    </section>

    <nav className="scenario-tabs" aria-label="Demo 场景">
      {scenarios.map((item, index) => <button key={item.id} className={`scenario-tab ${index === scenarioIndex ? 'active' : ''}`} onClick={() => chooseScenario(index)} disabled={busy}>
        <span>{item.emoji}</span>{item.title}
      </button>)}
    </nav>

    <section className="scenario-heading">
      <div><div className="scenario-title-line"><h2>{scenario.title}</h2><code>{scenario.code}</code></div><p>{scenario.desc}</p></div>
      <span className="question-count">{Object.keys(scenario.questions).length} 个 typed questions · 一次调用</span>
    </section>

    <section className="workspace-grid">
      <div className="input-column">
        <article className="panel input-panel">
          <div className="panel-head"><div><span className="panel-kicker">INPUT / STATE</span><h3>场景状态</h3></div><span className="sample-count">{scenario.samples.length} 个测试样例</span></div>
          <div className="sample-list" aria-label="测试样例">
            {scenario.samples.map((item, index) => <button key={item.name} className={index === sampleIndex ? 'selected' : ''} onClick={() => chooseSample(index)} disabled={busy}>{item.name}</button>)}
          </div>
          <label className="state-label" htmlFor="scenario-state">输入给模型的状态</label>
          <textarea id="scenario-state" value={stateText} onChange={(event) => { setStateText(event.target.value); setResult(null); }} disabled={busy} />
          <div className="input-foot"><span>{edited ? <><CircleHelp size={13} /> 自定义输入 · 不参与参考标签计分</> : <><ShieldCheck size={13} /> 当前场景上下文会随 state 一起发送</>}</span><span>{stateText.length} 字</span></div>
          <div className="context-card"><span>场景上下文</span><p>{Object.entries(scenario.context).map(([key, value]) => `${key}：${value}`).join(' · ')}</p></div>
          <div className="input-actions">
            <button className="primary-button" onClick={runOne} disabled={busy || !health || !stateText.trim()}>{busy ? <LoaderCircle size={16} className="spin" /> : <Play size={15} fill="currentColor" />}{busy ? '运行中…' : '运行当前样例'}</button>
            <button className="secondary-button" onClick={runScenario} disabled={busy || !health}><Activity size={15} /> 跑本场景 {scenario.samples.length} 条</button>
            {edited && <button className="icon-button reset-button" onClick={() => { setStateText(sample.text); setResult(null); }} title="恢复参考样例" aria-label="恢复参考样例"><RotateCcw size={15} /></button>}
          </div>
        </article>
        <details className="question-details"><summary><ChevronDown size={14} /> 查看本场景问题定义</summary><pre>{JSON.stringify(scenario.questions, null, 2)}</pre></details>
      </div>

      <div className="result-column">
        <article className="panel result-panel">
          <div className="panel-head result-header"><div><span className="panel-kicker">MODEL OUTPUT</span><h3>返回 / answers</h3></div><div className="result-head-right"><span className={`result-state ${currentResult ? 'finished' : error ? 'failed' : ''}`}>{currentResult ? '真实模型返回' : error ? '请求失败' : busy ? '请求中' : '等待调用'}</span></div></div>
          <div className="threshold-row"><div><Gauge size={14} /><span>动作参考线</span><small>{scenario.primary} · P(所选项) {actionConfidence == null ? '—' : percent(actionConfidence)}</small></div><label><span>复核阈值</span><b>{threshold.toFixed(2)}</b><input aria-label="动作复核阈值" type="range" min="0.3" max="0.95" step="0.01" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} /></label></div>
          {currentResult && actionAnswer && <div className={`action-line ${actionConfidence >= threshold ? 'action-ready' : 'action-review'}`}><span>{actionConfidence >= threshold ? '建议按策略继续' : '建议人工复核'}</span><small>仅按所选阈值由前端计算，不会触发真实操作；低置信度/未命中阈值时人工确认。</small></div>}
          {error && <div className="error-box"><AlertCircle size={17} /><div><b>推理没有完成</b><p>{error}</p><button onClick={checkHealth}>重新检查接口</button></div></div>}
          <div className="answers-list">{rows.map((row) => <AnswerCard row={row} key={row.id} />)}</div>
          <div className="result-toolbar"><div><SquareTerminal size={14} /><span>原始返回</span></div><button onClick={() => setShowRaw((current) => !current)} disabled={!currentResult}>{showRaw ? '收起 JSON' : '查看 JSON'} <ChevronDown size={13} className={showRaw ? 'rotate' : ''} /></button></div>
          {showRaw && currentResult && <pre className="raw-response">{JSON.stringify(currentResult.data, null, 2)}</pre>}
          <div className="result-meta"><span>样例 <b>{currentResult?.sampleName || (edited ? '自定义' : sample.name)}</b></span><span>耗时 <b>{currentResult ? `${currentResult.latency.toFixed(1)} ms` : '—'}</b></span><span>参考计分 <b>{currentResult ? (currentResult.reference ? '已计分' : '不计分') : '未运行'}</b></span></div>
        </article>
      </div>
    </section>

    {(busy || progress) && <div className={`progress-line ${busy ? 'active' : ''}`} role="status">{busy && <LoaderCircle size={14} className="spin" />}{progress}{busy && progress.includes('/') && <span>请求串行执行 · 保留已完成记录</span>}</div>}

    <section className="panel evaluation-panel">
      <div className="panel-head eval-head"><div><span className="panel-kicker">EVALUATION</span><h3>评测记录</h3></div><div className="eval-note">只计真实 API 返回；参考标签不会发送给模型。</div></div>
      <div className="eval-content">
        <div className="metric-grid">
          <div className="metric-card"><span>已运行样例</span><b>{records.length}</b><small>条真实推理</small></div>
          <div className="metric-card"><span>标签一致率</span><b>{totalScored ? `${(totalCorrect / totalScored * 100).toFixed(1)}%` : '—'}</b><small>{totalScored ? `${totalCorrect} / ${totalScored} 个问题` : '等待参考对照'}</small></div>
          <div className="metric-card"><span>平均耗时</span><b>{averageLatency === null ? '—' : `${averageLatency.toFixed(0)} ms`}</b><small>按样例计算</small></div>
          <div className="metric-card"><span>阈值线</span><b>{threshold.toFixed(2)}</b><small>前端策略参数</small></div>
        </div>
        <div className="record-table-wrap"><table><thead><tr><th>场景 / 样例</th><th>标签对照</th><th>延迟</th><th>状态</th></tr></thead><tbody>
          {records.length === 0 ? <tr><td colSpan="4" className="empty-record">还没有运行评测样例。选一个场景，点“运行当前样例”开始。</td></tr> : records.slice(0, 30).map((record) => <tr key={record.key}><td><b>{record.scenario}</b><small>{record.sample}</small></td><td>{record.total ? <span className={record.correct === record.total ? 'table-good' : 'table-mixed'}>{record.correct} / {record.total}</span> : '自定义 · 未计分'}</td><td>{record.latency.toFixed(1)} ms</td><td><CheckCircle2 size={15} className="table-check" /></td></tr>)}
        </tbody></table></div>
      </div>
    </section>

    <footer className="footer"><span><b>Laya 中文场景验证台</b> · 本地 FastAPI 推理 · checkpoint 通过只读卷挂载</span><span><ArrowDownToLine size={13} /> P(true) 和 confidence 是不同字段；请以独立中文验证集选择阈值。</span></footer>
  </main>;
}
