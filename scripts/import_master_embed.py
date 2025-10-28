from __future__ import annotations

from pathlib import Path
import sqlite3
import re


TEXT = r"""
資産の部
流動資産（現金・預金・売上債権など）
現金 GENKIN 100  
小口現金 KOGUCHI 101  
当座預金 TOZAYOK 110  
普通預金 FUTSUYO 115  
定期預金 TEIKIYO 124  
通知預金 TSUCHIYO 120  
定期積金 TEIKITSU 128  
別段預金 BETSUDAN 129  
郵便貯金 YUBIN 130  
受取手形 UKETORI 140  
不渡手形 FUWATARI 141  
売掛金 URIKAKE 142  
貸倒引当金（売）KASHIDAO 149  
有価証券 YUKASHO 150  
商品 SHOHIN 160  
仕掛品 SHIKAKAR 163  
貯蔵品 CHOZOHIN 165  
前渡金 MAEWATAS 170  
立替金 TATEKAE 171  
前払費用 MAEBARAI 175  
繰延税金資産(流) KURINOBE 190  
未収収益 MISHUSHU 174  
短期貸付金 TANKIKAS 173  
未収入金 MISHUNYU 172  
仮払金 KARIBARA 176  
預け金 AZUKEKIN 177  
仮払消費税等 KARIBARA 180  
貸倒引当金（他）KASHIDAO 199

固定資産（有形・無形・投資その他）
建物 TATEMONO 200  
附属設備 FUZOKUSE 201  
構築物 KOCHIKU 202  
機械装置 KIKAISO 203  
車両運搬具 SHARYOU 204  
工具器具備品 KOGUKIGU 205  
一括償却資産 IKKATSUS 208  
減価償却累計額 GENKARUI 209  
土地 TOCHI 210  
建設仮勘定 KENSETSU 211  
電話加入権 DENWA 220  
工業所有権 KOGYOSHO 222  
営業権 EIGYOKEN 223  
ソフトウェア SOFUTO 225  
投資有価証券 TOSHIYUK 240  
関係会社株式 KANKEIGA 249  
出資金 SHUSSHI 241  
関係会社出資金 KANKEIGA 250  
敷金 SHIKIKIN 242  
差入保証金 SASHIIRE 243  
長期貸付金 CHOKIKA 244  
長期固定性預金 CHOKIKO 245  
長期滞留債権 CHOKITA 246  
長期前払費用 CHOKIMA 247  
前払年金費用 MAEBARAI 253  
繰延税金資産(固) KURINOBE 248  
保険積立金 HOKENTSU 249  
預託金 YOTAKUKI 250  
貸倒引当金(投) KASHIDAO 252

繰延資産・諸口
創立費 SORITSU 260  
開業費 KAIGYOHI 261  
複合 FUKUGO 000  
未確定勘定 MIKAKUTE 001

負債の部
流動負債
支払手形 SHIHARAI 400  
買掛金 KAIKAKE 405  
短期借入金 TANKIKAR 410  
未払金 MIHARAIK 420  
未払費用 MIHARAIH 426  
未払配当金 MIHARAIH 421  
未払役員賞与 MIHARAIY 422  
未払法人税等 MIHARAIH 423  
未払消費税等 MIHARAIS 425  
繰延税金負債(流) KURINOBE 450  
前受金 MAEUKEKI 430  
預り金 AZUKARIK 427  
前受収益 MAEUKESH 431  
仮受金 KARIUKEK 428  
割引手形 WARIBIKI 432  
裏書手形 URAGAKI 435  
仮受消費税等 KARIUKES 440

固定負債
長期借入金 CHOKIKA 470  
役員借入金 YAKUINKA 471  
長期未払金 CHOKIMI 475  
繰延税金負債(固) KURINOBE 476  
退職給付引当金 TAISHOKU 490

純資産の部
資本金 SHIHONKI 500  
新株式申込証拠金 SHINKABU 510  
資本準備金 SHIHONJU 520  
資本金及び準備金減少差益 GENSHISA 525  
自己株式処分差額 JIKOKABU 526  
利益準備金 RIEKIJUN 530  
別途積立金 BETTOTSU 535  
繰越利益 KURIKOSH 540  
自己株式 JIKOKABU 550  
自己株式申込証拠金 JIKOKABU 560  
その他有価証券評価差額金 SONOTAYU 570  
繰延ヘッジ損益 KURINOBE 575  
土地再評価差額金 TOCHISAI 580  
新株予約権 SHINKABU 590

収益の部
売上高
売上高 URIAGE 700  
デザイン料収入 DEZAINRY 701  
制作料収入 SEISAKUR 702  
設計料収入 SEKKEIRY 703  
企画料収入 KIKAKURY 704  
受託業務収入 JUTAKUGY 705  
売上値引高 URIAGENE 707  
役務収益 EKIMUSHU 710

売上原価
期首商品棚卸高 KISHUSHO 720  
仕入高 SHIIREDA 725  
仕入戻し高 SHIIREMO 730  
期末商品棚卸高 KIMATSUS 737  
他勘定振替高（商）TAKANJO 739

販売費および一般管理費
役員報酬 YAKUINHO 740  
役員賞与 YAKUINSH 790  
給料手当 KYURYOTE 741  
雑給 ZAKKYU 744  
賞与 SHOYO 742  
退職金 TAISHOKU 743  
法定福利費 HOTEI 745  
福利厚生費 FUKURI 746  
退職給付費用 TAISHOKU 747  
採用教育費 SAIYOKYO 749  
外注費 GAICHUHI 750  
荷造運賃 NIZUKURI 751  
広告宣伝費 KOKOKU 752  
交際費 KOSAI 753  
会議費 KAIGI 754  
旅費交通費 RYOHI 755  
通信費 TSUSHIN 756  
販売手数料 HAMBAITE 757  
販売促進費 HAMBAISO 758  
消耗品費 SHOMOHIN 760  
事務用品費 JIMUYOHI 761  
修繕費 SHUZEN 762  
水道光熱費 SUIDO 763  
新聞図書費 SHIMBUN 764  
諸会費 SHOKAIHI 765  
支払手数料 SHIHARAI 766  
車両費 SHARYO 767  
地代家賃 CHIDAI 781  
賃借料 CHINSHAK 782  
リース料 RI-SU 768  
保険料 HOKEN 770  
租税公課 SOZEI 783  
支払報酬料 SHIHARAI 771  
寄付金 KIFUKIN 772  
研究開発費 KENKYU 773  
使用料 SHIYORYO 774  
制作費 SEISAKUH 775  
取材費 SHUZAIHI 776  
減価償却費 GENKASHO 780  
長期前払費用償却 CHOKIMAE 784  
繰延資産償却(販) KURINOBE 785  
貸倒損失(販) KASHIDAO 786  
貸倒引当金繰入額(販) KASHIDAO 787  
雑費 ZAPPI 789

営業外・特別損益など
支払利息 SHIHARAI 830  
貸倒損失(外) KASHIDAO 832  
有価証券売却損 YUKASHO 834  
繰延資産償却(外) KURINOBE 835  
貸倒引当金繰入額(外) KASHIDAO 836  
雑損失 ZATSUSON 846  
受取利息 UKETORIR 800  
受取配当金 UKETORIH 801  
仕入割引 SHIIREWA 802  
有価証券売却益 YUKASHO 803  
貸倒引当金戻入額 KASHIDAO 804  
雑収入 ZATSUSHU 816  
前期損益修正損 ZENKISON 933  
固定資産売却損 KOTEISHI 920  
固定資産除却損 KOTEISHI 921  
投資有価証券売却損 TOSHISHO 922  
前期損益修正益 ZENKISON 913  
固定資産売却益 KOTEISHI 900  
投資有価証券売却益 TOSHISHO 901  
法人税等 HOJINZEI 980  
法人税等調整額 HOJINZEI 981  
法人税、住民税及び事業税 HOJINZEI 982 
"""


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    db_path = base / 'data' / 'app.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        'CREATE TABLE IF NOT EXISTS account_master ('
        'id INTEGER PRIMARY KEY AUTOINCREMENT,'
        'code VARCHAR(32),'
        'name VARCHAR(128) UNIQUE,'
        'category VARCHAR(64),'
        'created_at TEXT DEFAULT CURRENT_TIMESTAMP,'
        'updated_at TEXT DEFAULT CURRENT_TIMESTAMP)'
    )
    ins = 0
    skip_kw = (
        '資産の部','流動資産','固定資産','繰延資産','諸口',
        '負債の部','流動負債','固定負債','純資産の部','収益の部',
        '売上高','売上原価','販売費および一般管理費','営業外','特別損益','など'
    )
    for ln in TEXT.splitlines():
        ln = ln.strip().replace('\u3000', ' ')
        if not ln or ln in skip_kw:
            continue
        m = re.match(r'^(.*?)[\s\u3000]*([A-Z][A-Z0-9\-]+)(?:\s+\d+)?\s*$', ln)
        if not m:
            continue
        name = m.group(1).strip()
        code = m.group(2).strip()
        # Debug show first few
        # print('PARSED:', name, code)
        cur.execute('INSERT OR IGNORE INTO account_master(name, code, category) VALUES (?,?,?)', (name, code, None))
        if cur.rowcount:
            ins += 1
    conn.commit()
    print(f'Imported master accounts: {ins}')


if __name__ == '__main__':
    main()
