import os
import httpx
import json
from typing import List, Dict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core imports
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ML Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# SIMPLE TF-IDF BASED RETRIEVER (No external downloads)
# ============================================================

class SimpleTfidfRetriever:
    """TF-IDF based retriever - works completely offline"""
    
    def __init__(self, documents: List[Document], k: int = 4):
        self.documents = documents
        self.k = k
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Fit vectorizer on documents
        texts = [doc.page_content for doc in documents]
        self.doc_vectors = self.vectorizer.fit_transform(texts)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents using TF-IDF similarity"""
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Get top k documents
        top_indices = similarities.argsort()[-self.k:][::-1]
        
        return [self.documents[i] for i in top_indices]


# ============================================================
# COMPLIANCE RAG SYSTEM
# ============================================================

class ComplianceRAGSystem:
    def __init__(self, api_key: str = None):
        """Initialize the system"""
        
        # LLM Setup with SSL bypass
        client = httpx.Client(verify=False)
        self.llm = ChatOpenAI(
            base_url="https://genailab.tcs.in",
            model="azure_ai/genailab-maas-DeepSeek-V3-0324",
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            http_client=client,
            temperature=0.1
        )
        
        print("✓ LLM initialized (no external model downloads needed)")
        
        self.documents = []
        self.retriever = None
        self.qa_chain = None
    
    def add_synthetic_data(self):
        """Add synthetic regulatory data"""
        docs_data = [
            {"content": """SEBI Circular SEBI/HO/MRD/2024/001
Section 4.2: Insider Trading Prohibition
No person with Unpublished Price Sensitive Information (UPSI) shall trade in securities when in possession of such information. 
Violation may result in penalties up to INR 25 crore or three times the profit made, whichever is higher.
Reference: SEBI (Prohibition of Insider Trading) Regulations, 2015
Additional Context: This regulation applies to all connected persons, immediate relatives, and entities under common control.
Trading window closures must be observed during sensitive periods.""", 
             "source": "SEBI_Insider_Trading_2024.pdf", "clause": "Section 4.2"},
            
            {"content": """FINRA Rule 5270: Front Running Prohibition
Members shall not trade ahead of customer orders in the same security. 
This includes trading by the member or persons associated with the member in advance of customer orders to take advantage of the anticipated price movement.
Penalty: Suspension, fine, or expulsion from membership.
Exceptions: Block trades and institutional facilitation with proper disclosure.""", 
             "source": "FINRA_Rule_5270.pdf", "clause": "Rule 5270"},
            
            {"content": """SEC Rule 10b-5: Anti-Fraud Provisions
It shall be unlawful to: (a) employ any device to defraud, (b) make untrue statements of material facts, (c) engage in any act that would operate as fraud in connection with the purchase or sale of securities.
Communications that mislead investors are strictly prohibited.
This includes pump-and-dump schemes, false rumors, and manipulative trading practices.""", 
             "source": "SEC_10b5_Regulations.pdf", "clause": "Rule 10b-5"},
            
            {"content": """Trading Limit Policy - Internal Guidelines
Single Trade Limit: USD 5,000,000 per transaction
Daily Aggregate Limit: USD 20,000,000 per trader
Concentration Limit: No more than 15% portfolio in single security
Breach Protocol: Immediate escalation to compliance officer
Requires pre-approval for trades exceeding 80% of limits.
Position limits apply to both long and short positions.
Real-time monitoring required for all trades.""", 
             "source": "Internal_Trading_Policy_2024.pdf", "clause": "Section 3.1-3.4"},
            
            {"content": """Market Abuse Regulation (MAR) - EU Directive
Article 15: Market Manipulation Prohibition
Prohibited activities include: pump-and-dump schemes, spoofing, layering, and dissemination of false information. 
Firms must have surveillance systems capable of detecting such patterns.
Administrative sanctions up to EUR 5,000,000 or 10% of annual turnover.
Criminal penalties may also apply including imprisonment up to 4 years.""", 
             "source": "EU_MAR_Guidelines.pdf", "clause": "Article 15"},
            
            {"content": """Best Execution Policy
Firms must take all sufficient steps to obtain the best possible result for clients.
Factors: price, costs, speed, likelihood of execution and settlement, size, nature.
Regular best execution reviews required quarterly.
Client consent needed for execution venues.
Order routing decisions must be documented.""",
             "source": "Best_Execution_Policy.pdf", "clause": "Section 2"},
            
            {"content": """Anti-Money Laundering (AML) Requirements
Suspicious Activity Reports (SARs) must be filed within 30 days.
Customer Due Diligence (CDD) required for all new accounts.
Enhanced Due Diligence (EDD) for high-risk customers.
Transaction monitoring systems must flag unusual patterns.
Record retention: 5 years minimum.""",
             "source": "AML_Compliance_Manual.pdf", "clause": "Chapter 3"},
            
            {"content": """Conflicts of Interest Management
Personal account dealing requires pre-clearance.
Chinese walls between research and trading.
Gift and entertainment limits: USD 250 per person per year.
Outside business activities must be disclosed.
Employee trading blackout periods during earnings.""",
             "source": "Conflicts_Policy.pdf", "clause": "Section 5"}
        ]
        
        for d in docs_data:
            self.documents.append(Document(
                page_content=d["content"],
                metadata={"source": d["source"], "clause": d["clause"], "source_type": "synthetic"}
            ))
        
        print(f"✓ Added {len(docs_data)} synthetic regulatory documents")
        return self
    
    def build_retriever(self):
        """Build retriever using TF-IDF (no external model needed)"""
        print("Building TF-IDF retriever (offline mode)...")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(self.documents)
        print(f"✓ Created {len(splits)} document chunks")
        
        # Use TF-IDF retriever (no downloads)
        self.retriever = SimpleTfidfRetriever(splits, k=6)
        
        print("✓ TF-IDF retriever ready (100% offline)")
        return self
    
    def setup_qa_chain(self):
        """Setup QA using LCEL"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a regulatory compliance advisor for capital markets. 
Answer the question using ONLY the provided context. You MUST:

1. Cite the exact source document and clause for every claim
2. Use this citation format: [Source: document_name, Clause: clause_number]
3. If the context doesn't contain enough information, respond with: 
   "I cannot provide a definitive answer based on available regulations. Please consult: [suggest relevant authority]"
4. Never make assumptions or use information not in the context
5. Provide confidence level: HIGH (explicitly stated), MEDIUM (implied), or LOW (unclear)

Context: {context}"""),
            ("human", "{question}")
        ])
        
        def format_docs(docs):
            return "\n\n".join([f"[Source: {d.metadata.get('source', 'Unknown')}, Clause: {d.metadata.get('clause', 'N/A')}]\n{d.page_content}" for d in docs])
        
        # LCEL Chain
        self.qa_chain = (
            {"context": lambda x: format_docs(self.retriever.get_relevant_documents(x["question"])), 
             "question": lambda x: x["question"]}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✓ QA chain configured with citation guardrails")
        return self
    
    def ask(self, question: str) -> Dict:
        """Ask a compliance question"""
        print(f"\n{'='*60}\nQUERY: {question}\n{'='*60}\n")
        
        # Get relevant docs
        docs = self.retriever.get_relevant_documents(question)
        
        # Get answer
        answer = self.qa_chain.invoke({"question": question})
        
        sources = [{"source": d.metadata.get('source'), "clause": d.metadata.get('clause'), 
                    "snippet": d.page_content[:200] + "..."} for d in docs[:4]]
        
        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"ANSWER:\n{answer}\n\n{'-'*60}\nSOURCES CITED: {len(sources)}")
        for i, s in enumerate(sources, 1):
            print(f"\n{i}. {s['source']} [{s['clause']}]")
            print(f"   Snippet: {s['snippet']}")
        print(f"{'='*60}\n")
        
        return result


# ============================================================
# SCENARIO SIMULATOR
# ============================================================

class ScenarioSimulator:
    def __init__(self, llm):
        self.llm = llm
        self.thresholds = {"trade_limit": 5000000, "concentration": 0.15, "daily_trades": 50}
    
    def simulate(self, description: str, current: Dict, proposed: Dict) -> Dict:
        print(f"\n{'='*60}\nSCENARIO SIMULATION\n{'='*60}\n")
        
        risk_score = 0
        violations = []
        
        if proposed.get('trade_size', 0) > self.thresholds['trade_limit']:
            risk_score += 0.3
            violations.append("Exceeds single trade limit ($5M)")
        if proposed.get('concentration', 0) > self.thresholds['concentration']:
            risk_score += 0.4
            violations.append("Exceeds concentration limit (15%)")
        if proposed.get('daily_trades', 0) > self.thresholds['daily_trades']:
            risk_score += 0.2
            violations.append("Exceeds daily trade frequency (50 trades)")
        
        level = "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.3 else "LOW"
        
        prompt = f"""Analyze this trading scenario for regulatory compliance risks:

Scenario: {description}
Current State: {json.dumps(current, indent=2)}
Proposed Changes: {json.dumps(proposed, indent=2)}
Risk Score: {risk_score} ({level})
Violations: {violations}

Provide:
1. Risk Assessment (Low/Medium/High)
2. Potential Regulation Breaches
3. Recommended Mitigations
4. Approval Requirements

Be specific and cite relevant rules."""
        
        response = self.llm.invoke(prompt)
        
        result = {
            "scenario": description,
            "risk_level": level,
            "breach_probability": round(min(risk_score * 1.2, 0.95), 2),
            "violations": violations,
            "assessment": response.content,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✓ Risk Level: {level}")
        print(f"✓ Breach Probability: {result['breach_probability']:.1%}")
        print(f"✓ Violations: {len(violations)}")
        print(f"{'='*60}\n")
        
        return result


# ============================================================
# CONDUCT RISK DETECTOR
# ============================================================

class ConductRiskDetector:
    def __init__(self, llm):
        self.llm = llm
        self.detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
    
    def generate_data(self, n=100) -> pd.DataFrame:
        """Generate synthetic trading data"""
        np.random.seed(42)
        df = pd.DataFrame({
            'trader_id': [f'T{i:03d}' for i in np.random.randint(1, 20, n)],
            'trade_size': np.random.lognormal(13, 1.5, n),
            'trade_count': np.random.poisson(10, n),
            'after_hours': np.random.binomial(5, 0.2, n),
        })
        
        # Add violations
        violations = np.random.choice(df.index, 10, replace=False)
        df.loc[violations, 'trade_size'] *= 3
        df.loc[violations, 'after_hours'] += 3
        df['is_violation'] = 0
        df.loc[violations, 'is_violation'] = 1
        
        # Messages
        normal_msgs = ["Normal trade execution", "Client order completed", "Portfolio rebalancing", "Standard market order"]
        suspicious_msgs = ["Got insider tip on XYZ stock", "Front run the client order", "Pump this stock before announcement", "Confidential merger info received"]
        df['message'] = [np.random.choice(suspicious_msgs) if v else np.random.choice(normal_msgs) 
                        for v in df['is_violation']]
        
        print(f"✓ Generated {n} trading records ({df['is_violation'].sum()} violations)")
        return df
    
    def train(self, df: pd.DataFrame):
        """Train anomaly detector"""
        X = df[['trade_size', 'trade_count', 'after_hours']]
        X_scaled = self.scaler.fit_transform(X)
        self.detector.fit(X_scaled)
        print("✓ Anomaly detector trained")
        return self
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect conduct risks"""
        X = df[['trade_size', 'trade_count', 'after_hours']]
        X_scaled = self.scaler.transform(X)
        
        df['is_flagged'] = (self.detector.predict(X_scaled) == -1).astype(int)
        df['anomaly_score'] = self.detector.score_samples(X_scaled)
        df['risk_score'] = -df['anomaly_score']
        
        # Normalize
        if df['risk_score'].max() != df['risk_score'].min():
            df['risk_score'] = (df['risk_score'] - df['risk_score'].min()) / (df['risk_score'].max() - df['risk_score'].min())
        
        print(f"✓ Detected {df['is_flagged'].sum()} flagged cases")
        return df
    
    def alerts(self, df: pd.DataFrame, n=3) -> List[Dict]:
        """Generate top alerts"""
        flagged = df[df['is_flagged'] == 1].nlargest(n, 'risk_score')
        
        alerts = []
        for idx, row in flagged.iterrows():
            alerts.append({
                "alert_id": f"ALERT-{idx:04d}",
                "trader_id": row['trader_id'],
                "risk_score": round(row['risk_score'], 3),
                "confidence": "HIGH" if row['risk_score'] > 0.7 else "MEDIUM",
                "trade_size": f"${row['trade_size']:,.0f}",
                "trade_count": int(row['trade_count']),
                "after_hours": int(row['after_hours']),
                "message": row['message'],
                "timestamp": datetime.now().isoformat()
            })
        
        return alerts
    
    def print_alerts(self, alerts: List[Dict]):
        """Print formatted alerts"""
        print(f"\n{'='*60}\nCONDUCT RISK ALERTS\n{'='*60}\n")
        for alert in alerts:
            print(f"Alert ID: {alert['alert_id']}")
            print(f"Trader: {alert['trader_id']}")
            print(f"Risk Score: {alert['risk_score']} ({alert['confidence']} confidence)")
            print(f"Trade Size: {alert['trade_size']}")
            print(f"Message: {alert['message']}")
            print(f"{'-'*60}\n")


# ============================================================
# EVALUATION METRICS
# ============================================================

class EvaluationMetrics:
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_prob=None) -> Dict:
        """Calculate ML metrics"""
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        metrics = {
            "Accuracy": round(accuracy, 3),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1-Score": round(f1, 3)
        }
        
        if y_prob is not None and len(set(y_true)) > 1:
            try:
                metrics["ROC-AUC"] = round(roc_auc_score(y_true, y_prob), 3)
            except:
                metrics["ROC-AUC"] = "N/A"
        
        return metrics


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "="*60)
    print("CAPITAL MARKETS COMPLIANCE & CONDUCT RISK ADVISOR")
    print("="*60 + "\n")
    
    # Initialize
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("⚠ WARNING: No API key found. Set OPENAI_API_KEY environment variable")
        print("Example: set OPENAI_API_KEY=your-key-here")
        return
    
    system = ComplianceRAGSystem(api_key=api_key)
    
    # Add data and build system
    system.add_synthetic_data()
    system.build_retriever()
    system.setup_qa_chain()
    
    # DEMO 1: Compliance Q&A
    print("\n" + "="*60)
    print("DEMO 1: COMPLIANCE ADVISORY Q&A")
    print("="*60 + "\n")
    
    questions = [
        "What are the regulations regarding insider trading in India?",
        "Can a trader execute trades ahead of client orders?",
        "What is the maximum single trade limit according to internal policy?"
    ]
    
    qa_responses = []
    for q in questions:
        response = system.ask(q)
        qa_responses.append(response)
    
    # DEMO 2: Scenario Simulation
    print("\n" + "="*60)
    print("DEMO 2: SCENARIO SIMULATOR")
    print("="*60 + "\n")
    
    simulator = ScenarioSimulator(system.llm)
    scenario_result = simulator.simulate(
        "Increase single trade limit from $5M to $10M",
        {"trade_limit": 5000000, "avg_daily_volume": 15000000, "historical_breaches": 2},
        {"trade_size": 10000000, "concentration": 0.20, "daily_trades": 45}
    )
    
    # DEMO 3: Conduct Risk Detection
    print("\n" + "="*60)
    print("DEMO 3: CONDUCT RISK DETECTION")
    print("="*60 + "\n")
    
    detector = ConductRiskDetector(system.llm)
    trading_data = detector.generate_data(n=100)
    detector.train(trading_data)
    risk_results = detector.detect(trading_data)
    alerts = detector.alerts(risk_results, n=3)
    detector.print_alerts(alerts)
    
    # DEMO 4: Evaluation Metrics
    print("\n" + "="*60)
    print("DEMO 4: EVALUATION METRICS")
    print("="*60 + "\n")
    
    evaluator = EvaluationMetrics()
    
    y_true = risk_results['is_violation'].values
    y_pred = risk_results['is_flagged'].values
    y_prob = risk_results['risk_score'].values
    
    ml_metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)
    print("ML Classification Metrics:")
    print(json.dumps(ml_metrics, indent=2))
    
    # Save outputs
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60 + "\n")
    
    with open('conduct_risk_alerts.json', 'w') as f:
        json.dump(alerts, f, indent=2)
    print("✓ Saved: conduct_risk_alerts.json")
    
    with open('scenario_simulation_report.json', 'w') as f:
        json.dump(scenario_result, f, indent=2)
    print("✓ Saved: scenario_simulation_report.json")
    
    risk_results.to_csv('conduct_risk_analysis.csv', index=False)
    print("✓ Saved: conduct_risk_analysis.csv")
    
    with open('qa_responses.json', 'w') as f:
        json.dump(qa_responses, f, indent=2)
    print("✓ Saved: qa_responses.json")
    
    evaluation_report = {
        "ml_metrics": ml_metrics,
        "timestamp": datetime.now().isoformat()
    }
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    print("✓ Saved: evaluation_metrics.json")
    
    print("\n" + "="*60)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
