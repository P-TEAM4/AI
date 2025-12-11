# AI/src/api/clips_api.py

import os
import sys
import shutil
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List, Dict
import requests
from dotenv import load_dotenv
from xgboost import XGBClassifier

# 모듈 import
from highlight_extractor import (
    extract_highlights_from_timeline,
    create_clip,
    get_top_highlights
)
from impact_highlight_integration import (
    enrich_highlights_with_impact,
    generate_match_summary
)

load_dotenv()

app = FastAPI(title="LoL Highlight Clips API with Impact Score", version="2.0.0")

# Impact Score 모델 로드
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "player_impact_model.json")
FEAT_PATH = os.path.join(MODEL_DIR, "feature_cols.json")

print(f"[INFO] Loading Impact Score model from {MODEL_PATH}")
impact_model = XGBClassifier()
impact_model.load_model(MODEL_PATH)

with open(FEAT_PATH, "r") as f:
    feature_cols = json.load(f)

print(f"[INFO] Impact Score model loaded with {len(feature_cols)} features")

RIOT_API_KEY = os.getenv("RIOT_API_KEY")
UPLOAD_DIR = "uploads"
CLIPS_DIR = "clips"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)


def get_timeline_data(match_id: str) -> dict:
    """Riot API로 타임라인 데이터 가져오기"""
    url = f"https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    params = {"api_key": RIOT_API_KEY}

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Riot API error: {response.text}"
        )

    return response.json()


def get_match_data(match_id: str) -> dict:
    """Riot API로 매치 데이터 가져오기"""
    url = f"https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}"
    params = {"api_key": RIOT_API_KEY}

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Riot API error: {response.text}"
        )

    return response.json()


@app.post("/clips/generate")
async def generate_highlight_clips(
    video: UploadFile = File(...),
    match_id: str = Form(...),
    game_name: str = Form(...),
    tag_line: str = Form(...),
    top_highlights: int = Form(5),
    top_mistakes: int = Form(3)
):
    """
    영상을 업로드하고 하이라이트 클립 생성

    Args:
        video: 게임 영상 파일
        match_id: 매치 ID (예: KR_7951354433)
        game_name: Riot ID 이름
        tag_line: Riot ID 태그
        top_highlights: 잘한 장면 개수
        top_mistakes: 못한 장면 개수

    Returns:
        생성된 클립 정보
    """
    try:
        # 1. PUUID 가져오기
        account_url = f"https://asia.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        params = {"api_key": RIOT_API_KEY}
        account_resp = requests.get(account_url, params=params)

        if account_resp.status_code != 200:
            raise HTTPException(status_code=404, detail="Player not found")

        puuid = account_resp.json()["puuid"]

        # 2. 영상 저장
        video_filename = f"{match_id}_{puuid}.mp4"
        video_path = os.path.join(UPLOAD_DIR, video_filename)

        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        print(f"[INFO] Saved video: {video_path}")

        # 3. 타임라인 데이터 가져오기
        timeline_data = get_timeline_data(match_id)
        match_data = get_match_data(match_id)

        # 4. 하이라이트 추출
        all_highlights = extract_highlights_from_timeline(timeline_data, puuid)

        if not all_highlights:
            raise HTTPException(status_code=404, detail="No highlights found for this player")

        # 4.5. Impact Score 모델과 연동하여 중요도 강화
        participant_id = None
        for idx, p in enumerate(timeline_data['metadata']['participants']):
            if p == puuid:
                participant_id = idx + 1
                break

        if participant_id:
            all_highlights = enrich_highlights_with_impact(
                all_highlights,
                timeline_data,
                match_data,
                participant_id,
                impact_model,
                feature_cols
            )

        # 5. 잘한 부분 / 못한 부분 분리
        highlight_clips = get_top_highlights(all_highlights, top_n=top_highlights, category='highlight')
        mistake_clips = get_top_highlights(all_highlights, top_n=top_mistakes, category='mistake')

        # 6. 클립 생성
        created_highlights = []
        created_mistakes = []

        for h in highlight_clips:
            clip_path = create_clip(video_path, h, output_dir=os.path.join(CLIPS_DIR, match_id))
            if clip_path:
                created_highlights.append({
                    "clip_path": clip_path,
                    "timestamp": h["timestamp"],
                    "type": h["type"],
                    "base_importance": h["importance"],
                    "impact_score": h.get("impact_score", 0),
                    "combined_importance": h.get("combined_importance", h["importance"]),
                    "description": h["description"],
                    "impact_description": h.get("impact_description", ""),
                    "details": h["details"]
                })

        for h in mistake_clips:
            clip_path = create_clip(video_path, h, output_dir=os.path.join(CLIPS_DIR, match_id))
            if clip_path:
                created_mistakes.append({
                    "clip_path": clip_path,
                    "timestamp": h["timestamp"],
                    "type": h["type"],
                    "base_importance": h["importance"],
                    "impact_score": h.get("impact_score", 0),
                    "combined_importance": h.get("combined_importance", h["importance"]),
                    "description": h["description"],
                    "impact_description": h.get("impact_description", ""),
                    "details": h["details"]
                })

        # 7. 매치 정보 추가
        player_info = None
        for participant in match_data['info']['participants']:
            if participant['puuid'] == puuid:
                player_info = {
                    "championName": participant['championName'],
                    "teamPosition": participant['teamPosition'],
                    "win": participant['win'],
                    "kills": participant['kills'],
                    "deaths": participant['deaths'],
                    "assists": participant['assists']
                }
                break

        # 7.5. 매치 요약 생성
        if participant_id:
            match_summary = generate_match_summary(all_highlights, match_data, participant_id)
        else:
            match_summary = None

        return {
            "match_id": match_id,
            "player": {
                "gameName": game_name,
                "tagLine": tag_line,
                "puuid": puuid
            },
            "match_info": player_info,
            "match_summary": match_summary,
            "highlights": created_highlights,
            "mistakes": created_mistakes,
            "video_path": video_path,
            "total_clips": len(created_highlights) + len(created_mistakes)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/clips/list/{match_id}")
def list_clips(match_id: str):
    """특정 매치의 생성된 클립 목록 조회"""
    clips_dir = os.path.join(CLIPS_DIR, match_id)

    if not os.path.exists(clips_dir):
        raise HTTPException(status_code=404, detail="No clips found for this match")

    clips = []
    for filename in os.listdir(clips_dir):
        if filename.endswith('.mp4'):
            clips.append({
                "filename": filename,
                "path": os.path.join(clips_dir, filename)
            })

    return {"match_id": match_id, "clips": clips, "total": len(clips)}


@app.delete("/clips/{match_id}")
def delete_clips(match_id: str):
    """특정 매치의 클립 삭제"""
    clips_dir = os.path.join(CLIPS_DIR, match_id)
    video_path = None

    # 업로드된 영상 찾기
    for filename in os.listdir(UPLOAD_DIR):
        if filename.startswith(match_id):
            video_path = os.path.join(UPLOAD_DIR, filename)
            break

    deleted_files = []

    # 클립 디렉토리 삭제
    if os.path.exists(clips_dir):
        shutil.rmtree(clips_dir)
        deleted_files.append(clips_dir)

    # 원본 영상 삭제
    if video_path and os.path.exists(video_path):
        os.remove(video_path)
        deleted_files.append(video_path)

    return {"deleted_files": deleted_files}


@app.post("/clips/test-timeline")
async def test_timeline_only(
    match_id: str = Form(...),
    game_name: str = Form(...),
    tag_line: str = Form(...),
    top_highlights: int = Form(5),
    top_mistakes: int = Form(3)
):
    """
    영상 없이 타임라인만으로 하이라이트 정보 테스트

    Args:
        match_id: 매치 ID (예: KR_7951354433)
        game_name: Riot ID 이름
        tag_line: Riot ID 태그
        top_highlights: 잘한 장면 개수
        top_mistakes: 못한 장면 개수

    Returns:
        하이라이트 정보 (클립 생성 없음)
    """
    try:
        # 1. PUUID 가져오기
        print(f"[DEBUG] Getting PUUID for {game_name}#{tag_line}")
        account_url = f"https://asia.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        params = {"api_key": RIOT_API_KEY}
        account_resp = requests.get(account_url, params=params)

        if account_resp.status_code != 200:
            print(f"[ERROR] Player not found: {account_resp.status_code}")
            raise HTTPException(status_code=404, detail="Player not found")

        puuid = account_resp.json()["puuid"]
        print(f"[DEBUG] Found PUUID: {puuid}")

        # 2. 타임라인 데이터 가져오기
        print(f"[DEBUG] Getting timeline for match: {match_id}")
        timeline_data = get_timeline_data(match_id)
        match_data = get_match_data(match_id)
        print(f"[DEBUG] Timeline and match data retrieved successfully")

        # 3. 하이라이트 추출
        print(f"[DEBUG] Extracting highlights for PUUID: {puuid}")
        print(f"[DEBUG] Timeline participants: {timeline_data['metadata']['participants']}")
        print(f"[DEBUG] Total frames: {len(timeline_data['info']['frames'])}")

        all_highlights = extract_highlights_from_timeline(timeline_data, puuid)

        print(f"[DEBUG] Extracted {len(all_highlights)} highlights")

        if not all_highlights:
            raise HTTPException(status_code=404, detail="No highlights found for this player")

        # 4. Impact Score 모델과 연동하여 중요도 강화
        participant_id = None
        for idx, p in enumerate(timeline_data['metadata']['participants']):
            if p == puuid:
                participant_id = idx + 1
                break

        if participant_id:
            all_highlights = enrich_highlights_with_impact(
                all_highlights,
                timeline_data,
                match_data,
                participant_id,
                impact_model,
                feature_cols
            )

        # 5. 잘한 부분 / 못한 부분 분리
        highlight_clips = get_top_highlights(all_highlights, top_n=top_highlights, category='highlight')
        mistake_clips = get_top_highlights(all_highlights, top_n=top_mistakes, category='mistake')

        # 6. 클립 정보 (영상 없이 타임스탬프만)
        highlights_info = []
        mistakes_info = []

        for h in highlight_clips:
            highlights_info.append({
                "timestamp": h["timestamp"],
                "type": h["type"],
                "base_importance": h["importance"],
                "impact_score": h.get("impact_score", 0),
                "combined_importance": h.get("combined_importance", h["importance"]),
                "description": h["description"],
                "impact_description": h.get("impact_description", ""),
                "details": h["details"]
            })

        for h in mistake_clips:
            mistakes_info.append({
                "timestamp": h["timestamp"],
                "type": h["type"],
                "base_importance": h["importance"],
                "impact_score": h.get("impact_score", 0),
                "combined_importance": h.get("combined_importance", h["importance"]),
                "description": h["description"],
                "impact_description": h.get("impact_description", ""),
                "details": h["details"]
            })

        # 7. 매치 정보 추가
        player_info = None
        for participant in match_data['info']['participants']:
            if participant['puuid'] == puuid:
                player_info = {
                    "championName": participant['championName'],
                    "teamPosition": participant['teamPosition'],
                    "win": participant['win'],
                    "kills": participant['kills'],
                    "deaths": participant['deaths'],
                    "assists": participant['assists']
                }
                break

        # 8. 매치 요약 생성
        if participant_id:
            match_summary = generate_match_summary(all_highlights, match_data, participant_id)
        else:
            match_summary = None

        return {
            "match_id": match_id,
            "player": {
                "gameName": game_name,
                "tagLine": tag_line,
                "puuid": puuid
            },
            "match_info": player_info,
            "match_summary": match_summary,
            "highlights": highlights_info,
            "mistakes": mistakes_info,
            "total_highlights": len(highlights_info) + len(mistakes_info)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/matches/{game_name}/{tag_line}")
def get_recent_matches(game_name: str, tag_line: str, count: int = 5):
    """
    플레이어의 최근 매치 ID 목록 가져오기

    Args:
        game_name: Riot ID 이름
        tag_line: Riot ID 태그
        count: 가져올 매치 개수 (기본 5개)

    Returns:
        최근 매치 ID 목록
    """
    try:
        # PUUID 가져오기
        account_url = f"https://asia.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        params = {"api_key": RIOT_API_KEY}
        account_resp = requests.get(account_url, params=params)

        if account_resp.status_code != 200:
            raise HTTPException(status_code=404, detail="Player not found")

        puuid = account_resp.json()["puuid"]

        # 최근 매치 목록 가져오기
        matches_url = f"https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"api_key": RIOT_API_KEY, "start": 0, "count": count}
        matches_resp = requests.get(matches_url, params=params)

        if matches_resp.status_code != 200:
            raise HTTPException(
                status_code=matches_resp.status_code,
                detail=f"Riot API error: {matches_resp.text}"
            )

        match_ids = matches_resp.json()

        return {
            "gameName": game_name,
            "tagLine": tag_line,
            "puuid": puuid,
            "match_ids": match_ids,
            "count": len(match_ids)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/")
def root():
    """API 정보"""
    return {
        "name": "LoL Highlight Clips API",
        "version": "1.0.0",
        "endpoints": [
            "POST /clips/generate - 영상 업로드 및 클립 생성",
            "POST /clips/test-timeline - 타임라인만으로 하이라이트 정보 테스트",
            "GET /matches/{game_name}/{tag_line} - 최근 매치 ID 목록 조회",
            "GET /clips/list/{match_id} - 클립 목록 조회",
            "DELETE /clips/{match_id} - 클립 삭제"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting Clips API server on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
