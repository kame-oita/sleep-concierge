"""
==============================================================================
クラスタリング結果画像 サムネイル一覧作成プログラム
==============================================================================

【プログラムの目的】
    クラスタリング処理によって振り分けられた画像群を、
    クラスタごとに「コンタクトシート（サムネイル一覧画像）」として
    まとめ、視覚的に確認しやすい形式で出力します。

【想定フォルダ構成】
    上位フォルダ（ROOT_DIR）
    ├── クラスタリング結果画像_Deep Learning活用ResNet50/   ← 手法フォルダ
    │   ├── cluster_0/                                      ← クラスタフォルダ
    │   │   ├── img001.jpg
    │   │   └── img002.jpg
    │   ├── cluster_1/
    │   │   └── ...
    │   └── ...
    ├── クラスタリング結果画像_定番のK-means/
    │   └── ...
    └── ...

【出力フォルダ構成】
    上位フォルダ（ROOT_DIR）
    └── サムネイル一覧/                                     ← 自動生成
        ├── クラスタリング結果画像_Deep Learning活用ResNet50/
        │   ├── cluster_0_sheet_01.jpg
        │   ├── cluster_0_sheet_02.jpg  ← 画像数が多い場合は複数シート
        │   ├── cluster_1_sheet_01.jpg
        │   └── ...
        └── クラスタリング結果画像_定番のK-means/
            └── ...

【必要ライブラリ】
    pip install Pillow

【使い方】
    1. 下記「設定セクション」の ROOT_DIR を実際のパスに変更する
    2. 必要に応じてサムネイルサイズや列数などを調整する
    3. python cluster_thumbnail_maker.py を実行する

【作成日】2026-05-25
==============================================================================
"""

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ==============================================================================
# ▼▼▼ 設定セクション（必要に応じて変更してください）▼▼▼
# ==============================================================================

# 上位フォルダのパス（Windowsパスはバックスラッシュをそのまま使用可能）
ROOT_DIR = Path(r"J:\20260218_クラスタリング結果")

# 処理対象とするクラスタ結果画像フォルダ名のリスト
# （空リストにすると ROOT_DIR 直下の全サブフォルダを自動検出します）
TARGET_METHOD_FOLDERS = [
    "クラスタリング結果画像_Deep Learning活用ResNet50",
    "クラスタリング結果画像_定番のK-means",
]

# 出力先フォルダ名（ROOT_DIR 直下に作成されます）
OUTPUT_FOLDER_NAME = "サムネイル一覧"

# 対応する画像拡張子
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# ---- サムネイルサイズ（ピクセル）----
THUMB_W = 320   # サムネイル1枚の幅
THUMB_H = 240   # サムネイル1枚の高さ

# ---- コンタクトシートのレイアウト ----
COLS = 4        # 横方向に並べる枚数
ROWS = 5        # 縦方向に並べる枚数（1シートあたりの最大行数）

# ---- 上部トリミング設定 ----
# True  : 画像の上部を16:9相当で切り出す（Webページのヒーロー画像などに適する）
# False : 画像全体を縮小してサムネイルに収める
TOP_CROP = False

# ---- シート装飾 ----
LABEL_HEIGHT   = 40    # ファイル名ラベルの高さ（ピクセル）
HEADER_HEIGHT  = 55    # シート上部のタイトル行の高さ（ピクセル）
BG_COLOR       = (245, 245, 245)   # シート背景色（R, G, B）
TITLE_COLOR    = (30, 30, 30)      # タイトル文字色
LABEL_COLOR    = (60, 60, 60)      # ファイル名ラベル文字色
BORDER_COLOR   = (200, 200, 200)   # セル境界線色
BORDER_WIDTH   = 1                 # セル境界線の太さ（ピクセル）

# ---- 出力画像品質 ----
OUTPUT_QUALITY = 88   # JPEG 保存品質（1〜95）

# ==============================================================================
# ▲▲▲ 設定セクションここまで ▲▲▲
# ==============================================================================


# 1シートに収める最大画像枚数
PER_SHEET = COLS * ROWS


def get_font(size: int):
    """
    日本語フォントの取得を試みる。
    取得できない場合は PIL のデフォルトフォントにフォールバックします。

    【Windows での日本語フォント優先順位】
    1. メイリオ (meiryo.ttc)
    2. MS ゴシック (msgothic.ttc)
    3. PIL デフォルトフォント（日本語は表示されない場合あり）
    """
    font_candidates = [
        # Windows 標準日本語フォント
        "meiryo.ttc",
        "msgothic.ttc",
        "YuGothM.ttc",
        # macOS 日本語フォント
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        # Linux 日本語フォント（要インストール）
        "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            continue

    # どのフォントも見つからない場合はデフォルトフォントを使用
    # ※デフォルトフォントは日本語を描画できないため文字化けすることがあります
    print("  [警告] 日本語フォントが見つかりません。ファイル名が文字化けする場合があります。")
    return ImageFont.load_default()


def collect_image_paths(folder: Path) -> list[Path]:
    """
    指定フォルダ内の画像ファイルを再帰的に収集して返します。

    Parameters
    ----------
    folder : Path
        検索対象フォルダ

    Returns
    -------
    list[Path]
        ファイル名でソートされた画像パスのリスト
    """
    paths = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    return sorted(paths, key=lambda p: p.name)


def open_image(path: Path) -> Image.Image:
    """
    画像ファイルを開き、RGB モードに変換して返します。
    EXIF 回転情報を自動的に適用します。

    Parameters
    ----------
    path : Path
        読み込む画像のパス

    Returns
    -------
    Image.Image
        RGB モードの PIL 画像オブジェクト
    """
    img = Image.open(path)

    # EXIF の回転情報があれば自動補正（撮影向きを正しく扱う）
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass  # EXIF 情報がない場合は無視

    return img.convert("RGB")


def make_thumbnail(img: Image.Image) -> Image.Image:
    """
    画像をサムネイルサイズ（THUMB_W × THUMB_H）に変換します。

    TOP_CROP = True の場合:
        画像の上部を目標アスペクト比で切り取ってからリサイズします。
        Webページのヒーロー画像など、上部に主要コンテンツがある場合に有効です。

    TOP_CROP = False の場合:
        アスペクト比を保ちながら全体を縮小し、余白を白で埋めます。

    Parameters
    ----------
    img : Image.Image
        元画像

    Returns
    -------
    Image.Image
        THUMB_W × THUMB_H サイズのサムネイル画像
    """
    w, h = img.size

    if TOP_CROP:
        # 目標アスペクト比に合わせて上部から切り出す
        target_ratio = THUMB_W / THUMB_H
        crop_h = min(h, int(w / target_ratio))
        img = img.crop((0, 0, w, crop_h))

    # アスペクト比を保ちながら最大 THUMB_W × THUMB_H に収まるようリサイズ
    img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)

    # 余白を背景色で埋めてキャンバスサイズを統一する
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), "white")
    offset_x = (THUMB_W - img.width) // 2
    offset_y = (THUMB_H - img.height) // 2
    canvas.paste(img, (offset_x, offset_y))
    return canvas


def truncate_label(text: str, font, max_width: int) -> str:
    """
    ラベル文字列が max_width ピクセルに収まるよう末尾を「...」で省略します。

    Parameters
    ----------
    text : str
        元のラベル文字列（ファイル名の拡張子なし部分）
    font : ImageFont
        使用するフォント
    max_width : int
        許容最大幅（ピクセル）

    Returns
    -------
    str
        必要に応じて省略したラベル文字列
    """
    # フォントの幅計算（PIL バージョンによって API が異なる）
    def text_width(s):
        try:
            return font.getlength(s)
        except AttributeError:
            return font.getsize(s)[0]

    if text_width(text) <= max_width:
        return text

    # 末尾から1文字ずつ削って省略記号を付加できる長さを探す
    while len(text) > 0:
        candidate = text + "..."
        if text_width(candidate) <= max_width:
            return candidate
        text = text[:-1]

    return "..."


def create_contact_sheet(
    image_paths: list[Path],
    sheet_index: int,
    total_sheets: int,
    cluster_name: str,
    method_name: str,
    font_title,
    font_label,
) -> Image.Image:
    """
    1枚のコンタクトシートを生成して返します。

    【シートの構成】
        ┌─────────────────────────────────────┐
        │ タイトル行（クラスタ名・シート番号・画像総数）│  HEADER_HEIGHT px
        ├────┬────┬────┬────┤
        │ 🖼 │ 🖼 │ 🖼 │ 🖼 │  THUMB_H px
        │ラベル│ラベル│ラベル│ラベル│  LABEL_HEIGHT px
        ├────┴────┴────┴────┤
        │         ...         │
        └─────────────────────────────────────┘

    Parameters
    ----------
    image_paths : list[Path]
        このシートに配置する画像パスのリスト（最大 PER_SHEET 枚）
    sheet_index : int
        シート番号（0 始まり）
    total_sheets : int
        このクラスタの総シート数
    cluster_name : str
        クラスタフォルダ名
    method_name : str
        手法フォルダ名
    font_title : ImageFont
        タイトル用フォント
    font_label : ImageFont
        ラベル用フォント

    Returns
    -------
    Image.Image
        生成されたコンタクトシート画像
    """
    cell_h = THUMB_H + LABEL_HEIGHT
    sheet_w = COLS * THUMB_W
    sheet_h = HEADER_HEIGHT + ROWS * cell_h

    # シートの背景を作成
    sheet = Image.new("RGB", (sheet_w, sheet_h), BG_COLOR)
    draw = ImageDraw.Draw(sheet)

    # ---- タイトル行の描画 ----
    title_line1 = f"[{method_name}]  クラスタ: {cluster_name}"
    title_line2 = (
        f"シート {sheet_index + 1} / {total_sheets}  "
        f"（このシート: {len(image_paths)} 枚）"
    )
    draw.text((10, 6), title_line1, fill=TITLE_COLOR, font=font_title)
    draw.text((10, 30), title_line2, fill=(100, 100, 100), font=font_label)

    # ---- 各セルにサムネイルを配置 ----
    for idx, img_path in enumerate(image_paths):
        row = idx // COLS
        col = idx % COLS
        cell_x = col * THUMB_W
        cell_y = HEADER_HEIGHT + row * cell_h

        # --- サムネイル画像の生成 ---
        try:
            img = open_image(img_path)
            thumb = make_thumbnail(img)
            sheet.paste(thumb, (cell_x, cell_y))
        except Exception as exc:
            # 画像読み込みに失敗した場合はエラー表示セルを描画
            draw.rectangle(
                [cell_x, cell_y, cell_x + THUMB_W - 1, cell_y + THUMB_H - 1],
                fill=(255, 220, 220),
            )
            draw.text(
                (cell_x + 5, cell_y + THUMB_H // 2 - 10),
                f"読込エラー:\n{img_path.name[:20]}",
                fill=(180, 0, 0),
                font=font_label,
            )
            print(f"      [警告] 画像スキップ: {img_path.name} / {exc}")

        # --- セル境界線の描画 ---
        if BORDER_WIDTH > 0:
            draw.rectangle(
                [
                    cell_x,
                    cell_y,
                    cell_x + THUMB_W - 1,
                    cell_y + THUMB_H - 1,
                ],
                outline=BORDER_COLOR,
                width=BORDER_WIDTH,
            )

        # --- ファイル名ラベルの描画 ---
        label_text = truncate_label(
            img_path.stem,  # 拡張子を除いたファイル名
            font_label,
            THUMB_W - 10,
        )
        draw.text(
            (cell_x + 5, cell_y + THUMB_H + 4),
            label_text,
            fill=LABEL_COLOR,
            font=font_label,
        )

    return sheet


def process_cluster(
    cluster_dir: Path,
    output_dir: Path,
    method_name: str,
    font_title,
    font_label,
) -> int:
    """
    1つのクラスタフォルダを処理し、コンタクトシートを保存します。

    Parameters
    ----------
    cluster_dir : Path
        処理するクラスタフォルダ
    output_dir : Path
        コンタクトシートの出力先フォルダ
    method_name : str
        手法フォルダ名（タイトル表示用）
    font_title : ImageFont
        タイトル用フォント
    font_label : ImageFont
        ラベル用フォント

    Returns
    -------
    int
        処理した画像の総枚数
    """
    cluster_name = cluster_dir.name

    # クラスタ内の画像ファイルを収集
    image_paths = collect_image_paths(cluster_dir)
    if not image_paths:
        print(f"    [スキップ] 画像が見つかりません: {cluster_name}")
        return 0

    total_images = len(image_paths)
    total_sheets = math.ceil(total_images / PER_SHEET)

    print(f"    クラスタ: {cluster_name}  画像数: {total_images}  シート数: {total_sheets}")

    # 必要なシート数分だけコンタクトシートを生成
    for sheet_idx in range(total_sheets):
        # このシートに配置する画像のスライス
        batch = image_paths[sheet_idx * PER_SHEET : (sheet_idx + 1) * PER_SHEET]

        sheet = create_contact_sheet(
            image_paths=batch,
            sheet_index=sheet_idx,
            total_sheets=total_sheets,
            cluster_name=cluster_name,
            method_name=method_name,
            font_title=font_title,
            font_label=font_label,
        )

        # 出力ファイル名: 「クラスタ名_sheet_01.jpg」形式
        output_filename = f"{cluster_name}_sheet_{sheet_idx + 1:02d}.jpg"
        output_path = output_dir / output_filename
        sheet.save(output_path, quality=OUTPUT_QUALITY)
        print(f"      保存: {output_path.name}")

    return total_images


def process_method_folder(
    method_dir: Path,
    output_base_dir: Path,
    font_title,
    font_label,
) -> dict:
    """
    1つの手法フォルダを処理します。
    直下のサブフォルダをクラスタとみなして順番に処理します。

    Parameters
    ----------
    method_dir : Path
        処理する手法フォルダ（例: クラスタリング結果画像_定番のK-means）
    output_base_dir : Path
        出力先の基底フォルダ（サムネイル一覧/）
    font_title : ImageFont
        タイトル用フォント
    font_label : ImageFont
        ラベル用フォント

    Returns
    -------
    dict
        処理結果のサマリ {"method": 手法名, "clusters": クラスタ数, "images": 総画像数}
    """
    method_name = method_dir.name
    print(f"\n  ▶ 手法フォルダ: {method_name}")

    # 手法ごとの出力フォルダを作成
    method_output_dir = output_base_dir / method_name
    method_output_dir.mkdir(parents=True, exist_ok=True)

    # 直下のサブフォルダをクラスタとして列挙（名前順にソート）
    cluster_dirs = sorted(
        [p for p in method_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )

    if not cluster_dirs:
        # サブフォルダがない場合は手法フォルダ直下の画像を1クラスタとして扱う
        print(f"    サブフォルダなし → フォルダ直下の画像を処理します")
        total = process_cluster(
            method_dir, method_output_dir, method_name, font_title, font_label
        )
        return {"method": method_name, "clusters": 1, "images": total}

    total_images = 0
    for cluster_dir in cluster_dirs:
        total_images += process_cluster(
            cluster_dir, method_output_dir, method_name, font_title, font_label
        )

    return {
        "method": method_name,
        "clusters": len(cluster_dirs),
        "images": total_images,
    }


def main():
    """
    メイン処理。
    設定に従って全手法フォルダのコンタクトシートを生成します。
    """
    print("=" * 60)
    print("クラスタリング結果画像 サムネイル一覧作成プログラム")
    print("=" * 60)

    # ---- 入力フォルダの存在確認 ----
    if not ROOT_DIR.exists():
        print(f"\n[エラー] 上位フォルダが見つかりません: {ROOT_DIR}")
        print("スクリプト冒頭の ROOT_DIR を正しいパスに変更してください。")
        sys.exit(1)

    print(f"\n上位フォルダ: {ROOT_DIR}")

    # ---- 処理対象フォルダの決定 ----
    if TARGET_METHOD_FOLDERS:
        # 設定で指定されたフォルダのみを処理
        method_dirs = []
        for folder_name in TARGET_METHOD_FOLDERS:
            folder_path = ROOT_DIR / folder_name
            if folder_path.exists() and folder_path.is_dir():
                method_dirs.append(folder_path)
            else:
                print(f"  [警告] 指定フォルダが見つかりません（スキップ）: {folder_path}")
    else:
        # ROOT_DIR 直下のすべてのサブフォルダを自動検出
        method_dirs = sorted(
            [p for p in ROOT_DIR.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )
        print(f"対象フォルダを自動検出: {len(method_dirs)} 件")

    if not method_dirs:
        print("\n[エラー] 処理対象のフォルダが見つかりませんでした。")
        sys.exit(1)

    # ---- 出力フォルダの作成 ----
    output_base_dir = ROOT_DIR / OUTPUT_FOLDER_NAME
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"出力先: {output_base_dir}")

    # ---- フォントの準備 ----
    # タイトル用（やや大きめ）とラベル用（小さめ）の2種類を用意
    font_title = get_font(16)
    font_label = get_font(12)

    # ---- 各手法フォルダを順番に処理 ----
    summaries = []
    for method_dir in method_dirs:
        summary = process_method_folder(
            method_dir, output_base_dir, font_title, font_label
        )
        summaries.append(summary)

    # ---- 処理結果のサマリ表示 ----
    print("\n" + "=" * 60)
    print("処理完了サマリ")
    print("=" * 60)
    total_images_all = 0
    for s in summaries:
        print(
            f"  {s['method']}\n"
            f"    クラスタ数: {s['clusters']}  総画像数: {s['images']}"
        )
        total_images_all += s["images"]

    print(f"\n全処理画像数合計: {total_images_all} 枚")
    print(f"出力フォルダ: {output_base_dir}")
    print("\n処理が完了しました。")


if __name__ == "__main__":
    main()
