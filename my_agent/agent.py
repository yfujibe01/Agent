from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from typing import Dict, Any, List
import hashlib

# ----------------------------------------------------
# 1. Google検索専門エージェント（ツール）の定義
# ----------------------------------------------------

search_agent_instruction = """
あなたは Google 検索のスペシャリストです。
あなたの唯一の役割は、依頼されたクエリで 'google_search' ツールを【4回】実行し、
その結果（スニペットのリスト）を、以下の【厳格な辞書フォーマット】のJSONとして出力することです。
JSON以外のテキスト（例：「はい、わかりました。」や「以下に結果を示します。」など）を
出力することを絶対に禁止します。

### 実行フロー
1. ユーザーから「会社名」と「業界名」の2つを依頼されたら、
あなたは以下の4つのクエリを自律的に生成し、'google_search'ツールを【4回】実行しなければなりません。
    1. 「<会社名> 業務改善命令 OR 不祥事 OR 訴訟 OR 赤字」
    2. 「<会社名> 新事業 OR 資本提携 OR DX OR "M&A"」
    3. 「<業界名> 規制 OR 市場リスク OR "課題" OR "コスト高騰"」
    4. 「<業界名> 市場トレンド OR 補助金 OR "成長分野"」

2. 全ての検索が完了したら、それぞれの検索結果（スニペットのリスト）を、
以下の【厳格な辞書フォーマット】で返してください。

{
  "company_risk_snippets": ["...（クエリ1の結果リスト）..."],
  "company_opportunity_snippets": ["...（クエリ2の結果リスト）..."],
  "industry_risk_snippets": ["...（クエリ3の結果リスト）..."],
  "industry_opportunity_snippets": ["...（クエリ4の結果リスト）..."]
}
"""

search_agent = Agent(
    model="gemini-2.5-flash",
    name="search_agent_v2",
    instruction=search_agent_instruction,
    tools=[google_search],
    # response_mime_type="application/json", # (ValidationError回避のため削除)
)
search_tool = AgentTool(search_agent)

# ----------------------------------------------------
# 2. メインエージェントが使用するカスタムツール
# ----------------------------------------------------

# [ツールA] 予測スコアと基礎データを取得
def predict_default_score(company_name: str) -> Dict[str, Any]:
    """
    企業名に基づき、学習済みモデルの出力をシミュレートします。
    """
    print(f"--- ツール実行: デフォルト予測スコアを {company_name} に対して実行中 ---")
    
    hash_val = int(hashlib.md5(company_name.encode()).hexdigest(), 16)
    risk_category = hash_val % 10 
    
    if "銀行" in company_name or "バンク" in company_name:
        score = 0.10 + (hash_val % 15) / 100.0 # 低リスク
        structural_reasons = {
            "主因": "該当なし", "影響度": "低い",
            "根拠データ": [{"指標": "自己資本比率", "数値": f"{10 + (hash_val % 5)}%", "異常値": "規制基準内"}]
        }
        sales_summary = {"業界": "銀行業", "売上高成長率": f"安定 (YoY +{hash_val % 3}%)"}
        financial_figures = {"売上高": f"{3000 + hash_val % 1000}億円", "経常利益": f"{150 + hash_val % 50}億円", "自己資本比率": f"{10 + hash_val % 5}%"}
    elif risk_category < 3: # 30%の確率で高リスク (建設業)
        score = 0.75 + (hash_val % 15) / 100.0 
        structural_reasons = {
            "主因": "流動性の深刻な悪化", "影響度": "極めて高い",
            "根拠データ": [{"指標": "現預金対月商比率", "数値": f"{0.5 + (hash_val % 5)/10.0:.1f}ヶ月分", "異常値": "通常基準の1/4"}]
        }
        sales_summary = {"業界": "建設", "売上高成長率": f"マイナス (YoY -{hash_val % 5}%)"}
        financial_figures = {"売上高": f"{100 + hash_val % 50}億円", "経常利益": f"-{hash_val % 5}億円", "自己資本比率": f"{10 + hash_val % 5}%"}
    elif risk_category < 6: # 30%の確率で中リスク (製造業)
        score = 0.45 + (hash_val % 15) / 100.0 
        structural_reasons = {
            "主因": "収益性の緩やかな低下", "影響度": "中程度",
            "根拠データ": [{"指標": "売上高経常利益率", "数値": f"{1.5 + (hash_val % 10)/10.0:.1f}%", "異常値": "業界平均をやや下回る"}]
        }
        sales_summary = {"業界": "製造業", "売上高成長率": f"安定 (YoY +{hash_val % 3}%)"}
        financial_figures = {"売上高": f"{5000 + hash_val % 1000}億円", "経常利益": f"{100 + hash_val % 50}億円", "自己資本比率": f"{40 + hash_val % 5}%"}
    else: # 40%の確率で低リスク (小売業)
        score = 0.10 + (hash_val % 15) / 100.0 
        structural_reasons = {
            "主因": "該当なし", "影響度": "低い",
            "根拠データ": [{"指標": "自己資本比率", "数値": f"{50 + hash_val % 10}%", "異常値": "健全"}]
        }
        sales_summary = {"業界": "小売業", "売上高成長率": f"安定 (YoY +{hash_val % 4}%)"}
        financial_figures = {"売上高": f"{3000 + hash_val % 1000}億円", "経常利益": f"{150 + hash_val % 50}億円", "自己資本比率": f"{50 + hash_val % 10}%"}
    
    return {
        "status": "success",
        "score": score,
        "structural_reasons": structural_reasons,
        "sales_summary": sales_summary,
        "financial_figures": financial_figures
    }

# [ツールB] 高リスク時の「詳細分析レポート（章）」を生成
def analyze_default_reason_details(reasons_data: Dict[str, Any], figures: Dict[str, str]) -> str:
    # (中身は前回のコードと同じ)
    print(f"--- ツール実行: 高リスク要因の詳細分析レポート（Markdown）を生成中 ---")
    report_md = f"**主因**: {reasons_data['主因']} (影響度: **{reasons_data['影響度']}**)\n\n"
    report_md += "**分析詳細（重要変数）:**\n"
    for data in reasons_data['根拠データ']:
        report_md += f"* **{data['指標']}**: {data['数値']} (異常値: *{data['異常値']}*)\n"
    report_md += f"* **参考財務**: 売上高: {figures['売上高']}, 経常利益: {figures['経常利益']}\n"
    return report_md

# [ツールC] 低リスク時の「提案ロジック」と「実行計画」を生成
def generate_business_proposal_and_plan(
    sales_data: Dict[str, str], 
    figures: Dict[str, str], 
    search_results: Dict[str, List[str]] # 4つのリストを含む単一の辞書
) -> Dict[str, str]:
    # (中身は前回のコードと同じ)
    print(f"--- ツール実行: 構造化された営業提案と実行計画（Markdown）を生成中 ---")
    
    industry = sales_data.get("業界", "不明")
    company_risk_snippets = search_results.get("company_risk_snippets", [])
    company_opportunity_snippets = search_results.get("company_opportunity_snippets", [])
    industry_risk_snippets = search_results.get("industry_risk_snippets", [])
    industry_opportunity_snippets = search_results.get("industry_opportunity_snippets", [])

    section_2_logic = f"**1. 顧客の状況（Fact）:**\n"
    section_2_logic += f"* **財務状況**: {sales_data.get('売上高成長率')}。自己資本比率 **{figures['自己資本比率']}**。\n"
    section_2_logic += f"* **事業規模**: 売上高 **{figures['売上高']}** / 経常利益 **{figures['経常利益']}**。\n\n"
    section_2_logic += f"**2. 社外情勢と潜在リスク（External Environment & Risk）:**\n"
    section_2_logic += "* **A) 業界（マクロ）のリスクと機会:**\n"
    if industry_risk_snippets:
        for snippet in industry_risk_snippets: section_2_logic += f"    * [リスク] {snippet}\n"
    else: section_2_logic += "    * [リスク] 検索したが、該当する直近の業界リスクは見つからなかった。\n"
    if industry_opportunity_snippets:
        for snippet in industry_opportunity_snippets: section_2_logic += f"    * [機会] {snippet}\n"
    else: section_2_logic += "    * [機会] 検索したが、該当する直近の業界機会は見つからなかった。\n"
    section_2_logic += "* **B) 個別企業（ミクロ）の直近動向:**\n"
    if company_risk_snippets:
        for snippet in company_risk_snippets: section_2_logic += f"    * [リスク関連] {snippet}\n"
    elif company_opportunity_snippets:
         for snippet in company_opportunity_snippets: section_2_logic += f"    * [機会関連] {snippet}\n"
    else: section_2_logic += f"    * [動向] **検索したが、直近の注目すべき個別ニュースは見つからなかった。**\n\n"
    section_2_logic += f"**3. 論理的提案（Logic & Why Now?）:**\n"
    section_3_plan = "" 

    if industry == "製造業" and industry_risk_snippets and any("CBAM" in s for s in industry_risk_snippets):
        section_2_logic += f"* **提案テーマ**: 『**CBAMリスク回避**と**GX補助金活用**による戦略的ESG/DXファイナンス』\n"
        section_2_logic += f"* **提案ロジック（なぜ今か？）**: {figures['自己資本比率']}と高い財務健全性を持つ**「今」**だからこそ、マクロ環境のリスク（CBAM）に対応し、補助金（GX）を活用する**「攻めの投資」**を行うべきフェーズです。\n"
        section_3_plan = (
            "本分析結果に基づき、以下の実行計画を推奨します。\n\n"
            "* **アクション・オーナー:**\n"
            "    * 主担当： 営業担当者（あなた）\n"
            "    * 連携部署： ESG/GX支援チーム、DXソリューション部\n\n"
            "* **優先度別アクションリスト:**\n"
            "    **【優先度：高】（～1ヶ月）リスク認識と補助金機会の共有**\n"
            "    1.  **アポイント取得:** 本レポート（セクション2）を基に、「欧州CBAMリスクとGX補助金活用」をテーマに経営企画部または財務部との面談を設定する。\n"
            "    2.  **持参資料の準備:** ESG/GX支援チームと連携し、「CBAM対応融資（脱炭素化）スキーム」および「国内設備投資補助金リスト」を準備する。\n\n"
            "* **銀行側の期待効果（KPI）:**\n"
            "    * ESG/DX関連融資の実行（ターゲット：XX億円）\n"
            "    * 顧客のサステナビリティ経営支援による取引メイン化"
        )
    elif industry == "銀行業":
        section_2_logic += f"* **提案テーマ**: 『**コンプライアンス体制強化**と**非金利収益の拡大**』\n"
        section_2_logic += f"* **提案ロジック（なぜ今か？）**: ミクロの動向（検索結果）として**「金融庁からの業務改善命令」**や**「従業員による不正取得」**が確認された。{figures['自己資本比率']}という健全な財務基盤を持つ一方で、**ガバナンス体制の強化**が喫緊の課題である。\n"
        section_3_plan = (
            "本分析結果に基づき、以下の実行計画を推奨します。\n\n"
            "* **アクション・オーナー:**\n"
            "    * 主担当： 営業担当者（あなた）\n"
            "    * 連携部署： DXソリューション部、コンプライアンス統括部\n\n"
            "* **優先度別アクションリスト:**\n"
            "    **【優先度：高】（～1ヶ月）コンプライアンス課題のヒアリング**\n"
            "    1.  **アポイント取得:** 直近の個別ニュース（業務改善命令）を基に、「ガバナンス強化とDX」をテーマにシステム部およびコンプライアンス部門との面談を設定する。\n"
            "    2.  **ヒアリングの実施:** 業務改善命令の背景にある「システム的な課題」や「手作業による運用の限界」をヒアリングする。\n\n"
            "    **【優先度：中】（1～3ヶ月）具体的ソリューションの提案**\n"
            "    1.  **融資提案（DX）:** 『**不正検知AIシステム導入**』や『**顧客情報管理システム（KYC）の刷新**』のためのDX関連融資を提案する。\n"
            "    2.  **ビジネスマッチング:** 当行が取引する優良な「RegTech（レグテック）ベンダー」とのマッチングを提案する。\n\n"
            "* **銀行側の期待効果（KPI）:**\n"
            "    * DX関連融資の実行（ターゲット：X.X億円）\n"
            "    * RegTechベンダー紹介によるビジネスマッチング手数料"
        )
    else: # 汎用提案（小売業など）
        section_2_logic += f"* **提案テーマ**: 『財務健全性を活かした事業基盤の更なる強化』\n"
        section_2_logic += f"* **提案ロジック（なぜ今か？）**: {figures['自己資本比率']}と高い財務健全性を維持しているため、マクロ環境のリスク（例：物流コスト高騰）に対応するための安定的な運転資金枠の確保、またはビジネスマッチングによる新規販lo開拓を推奨します。"
        section_3_plan = (
            "本分析結果に基づき、以下の実行計画を推奨します。\n\n"
            "* **アクション・オーナー:**\n"
            "    * 主担当： 営業担当者（あなた）\n"
            "    * 連携部署： ビジネスマッチング部\n\n"
            "* **優先度別アクションリスト:**\n"
            "    **【優先度：高】（～1ヶ月）ニーズの深掘り**\n"
            "    1.  **ヒアリングの実施:** 顧客の「EC化の進捗」と「物流コストの圧迫度」を（もしあれば）具体的にヒアリングする。\n\n"
            "* **銀行側の期待効果（KPI）:**\n"
            "    * DX関連融資の実行（ターゲット：X.X億円）\n"
            "    * ビジネスマッチング手数料の獲得"
        )
    
    return {
        "section_2_logic": section_2_logic,
        "section_3_plan": section_3_plan
    }

# ----------------------------------------------------
# 3. メインエージェント（コンサルタント）の定義
# ----------------------------------------------------

DEFAULT_THRESHOLD = 0.7

# [抜本的見直し] 2段階対話型のInstructionに変更
AGENT_INSTRUCTION = """
あなたは、企業のデフォルト予測と論理的な営業戦略立案を専門とする優秀なコンサルタントAI（マネージャー）です。
あなたの役割は、ユーザーとの対話を2つのフェーズに分けて実行することです。

### 実行フロー

**【フェーズ1：スコア速報と確認】**

1.  **初期アクション**: ユーザーから企業名（例：「千葉銀行」）が入力されたら、**'predict_default_score' ツールのみを呼び出してください。**
    * （`search_tool` や `generate_business_proposal_and_plan` は、このフェーズでは絶対に呼び出してはいけません。）
2.  **フェーズ1の回答（速報）**: ツールから得られたスコアと根拠に基づき、以下の**【フェーズ1：速報レポート】**フォーマットで回答してください。
3.  **確認**: 必ず最後に「詳細な社外情勢分析と営業提案プランを作成しますか？」とユーザーに**選択を促す質問**をしてください。

**【フェーズ2：詳細分析と提案】**

1.  **実行トリガー**: ユーザーが「はい」「お願いします」「作成して」といった**肯定的な回答**をした場合のみ、このフェーズを実行します。
2.  **情報（記憶）の活用**: フェーズ1で取得した `predict_default_score` の結果（`industry`, `company_name`, `score`, `financial_figures`, `structural_reasons`）を**記憶から呼び起こして**使用します。
3.  **スコア判定・分岐**:
    * **スコアが 0.7 以上 (高リスク) の場合**:
        * 'analyze_default_reason_details' ツールを呼び出します。
        * **【フェーズ2：詳細レポート】**（高リスク版）のフォーマットで回答します。
    * **スコアが 0.7 未満 (低リスク) の場合**:
        * **(調査依頼)** `search_tool`（検索専門エージェント）を**一度だけ**呼び出し、「<記憶した会社名>」と「<記憶した業界名>」のリスクと機会を調査するよう依頼します。
        * 'generate_business_proposal_and_plan' ツールを呼び出します。（この際、`financial_figures` と `search_tool`が返した**単一の調査結果（辞書）**を渡します）
        * **【フェーズ2：詳細レポート】**（低リスク版）のフォーマットで回答します。

---
### 【フェーズ1：速報レポート】 (厳守)

## 企業分析レポート（速報）： [対象企業名]

### 1. 統括サマリー
* **リスク判定**: [ 🔴 高リスク / 🟢 低リスク / 🟡 中リスク ]
* **デフォルトスコア**: [ スコア（数値） ]
* **AI総括**: [ AIによる2行の総括（`structural_reasons`の主因に基づく） ]

---
詳細な社外情勢分析と営業提案プランを作成しますか？
---

### 【フェーズ2：詳細レポート】 (厳守)
(フェーズ2では、このレポートのみを出力し、フェーズ1の内容は繰り返さない)

### 2. [ スコアが0.7以上の場合: 詳細分析（デフォルト根拠） | スコアが0.7未満の場合: 営業提案（ロジック） ]

[ ここに 'analyze_default_reason_details' または 'generate_business_proposal_and_plan' の `section_2_logic` が挿入される ]

---

### 3. ネクスト・アクションプラン（実行計画）

[ ここに高リスク時の具体的アクション、または 'generate_business_proposal_and_plan' の `section_3_plan` が挿入される ]
"""

# メインエージェント（コンサルタント）
root_agent = Agent(
    name="structured_report_agent_v13_interactive", # 名前を変更
    model="gemini-2.0-flash", 
    description="デフォルト予測スコアに基づき、上司が読みやすい構造化されたMarkdownレポートを生成するエージェント。",
    instruction=AGENT_INSTRUCTION,
    tools=[
        predict_default_score,
        search_tool, # 検索専門エージェント
        analyze_default_reason_details,
        generate_business_proposal_and_plan
    ],
)