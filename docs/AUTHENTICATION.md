# 인증 전략 가이드

## 아키텍처 개요

```
┌─────────────┐
│   사용자     │
└──────┬──────┘
       │ 소셜 로그인 (OAuth2)
       ▼
┌─────────────────────────────────┐
│     Spring Boot Backend         │
│  ┌──────────────────────────┐   │
│  │  Spring Security         │   │
│  │  + OAuth2 Client         │   │
│  ├──────────────────────────┤   │
│  │  JWT 토큰 발급           │   │
│  └──────────────────────────┘   │
└────────┬────────────────────────┘
         │ JWT Token in Header
         ▼
┌─────────────────────────────────┐
│     FastAPI AI Service          │
│  ┌──────────────────────────┐   │
│  │  JWT 검증 (선택사항)      │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
```

## 옵션 1: AI 서비스 인증 불필요 (권장)

### 이유
- AI 서비스는 **내부 서비스**로만 동작
- Spring Boot만 외부(사용자)에 노출
- AI 서비스는 Spring Boot에서만 호출

### 구조
```
Internet
    │
    │ HTTPS
    ▼
┌─────────────────┐
│  Spring Boot    │ ← 여기서만 인증 처리
│  (Public)       │
└────────┬────────┘
         │ HTTP (내부 네트워크)
         ▼
┌─────────────────┐
│  AI Service     │ ← 인증 불필요
│  (Private)      │
└─────────────────┘
```

### Spring Boot 설정

#### 1. 소셜 로그인 구현 (Spring Security + OAuth2)

```java
// SecurityConfig.java
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .csrf(csrf -> csrf.disable())
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/", "/login/**", "/error").permitAll()
                .anyRequest().authenticated()
            )
            .oauth2Login(oauth2 -> oauth2
                .loginPage("/login")
                .defaultSuccessUrl("/home")
                .userInfoEndpoint(userInfo -> userInfo
                    .userService(customOAuth2UserService)
                )
            )
            .addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }
}
```

#### 2. JWT 토큰 발급

```java
// JwtTokenProvider.java
@Component
public class JwtTokenProvider {

    @Value("${jwt.secret}")
    private String secretKey;

    @Value("${jwt.expiration}")
    private Long expiration;

    public String generateToken(String userId, String email) {
        Date now = new Date();
        Date expiryDate = new Date(now.getTime() + expiration);

        return Jwts.builder()
            .setSubject(userId)
            .claim("email", email)
            .setIssuedAt(now)
            .setExpiration(expiryDate)
            .signWith(SignatureAlgorithm.HS512, secretKey)
            .compact();
    }

    public boolean validateToken(String token) {
        try {
            Jwts.parser().setSigningKey(secretKey).parseClaimsJws(token);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    public String getUserIdFromToken(String token) {
        Claims claims = Jwts.parser()
            .setSigningKey(secretKey)
            .parseClaimsJws(token)
            .getBody();

        return claims.getSubject();
    }
}
```

#### 3. AI 서비스 호출 (인증 없이)

```java
// AIAnalysisService.java
@Service
public class AIAnalysisService {

    @Value("${ai.service.url}")
    private String aiServiceUrl;

    private final WebClient webClient;

    public AIAnalysisService(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder
            .baseUrl(aiServiceUrl)
            .build();
    }

    public Mono<MatchAnalysisResult> analyzeMatch(String matchId, String summonerName) {
        // AI 서비스는 내부 호출이므로 별도 인증 불필요
        return webClient.post()
            .uri("/api/v1/analyze/match")
            .bodyValue(new MatchRequest(matchId, summonerName, "KR1"))
            .retrieve()
            .bodyToMono(MatchAnalysisResult.class);
    }
}
```

### FastAPI 설정 (인증 불필요)

```python
# src/api/routes.py

@app.post("/api/v1/analyze/match")
async def analyze_match(request: MatchRequest):
    """
    매치 분석 - 인증 불필요
    Spring Boot에서만 호출 가능 (내부 네트워크)
    """
    result = match_analyzer.analyze_match(
        match_id=request.match_id,
        summoner_name=request.summoner_name,
        tag_line=request.tag_line,
    )
    return result
```

### 네트워크 레벨 보안

```yaml
# docker-compose.yml
version: '3.8'

services:
  spring-boot:
    image: spring-boot-app:latest
    ports:
      - "8080:8080"  # 외부 노출
    networks:
      - frontend
      - backend

  ai-service:
    image: ai-service:latest
    # 포트를 외부에 노출하지 않음 (내부 네트워크만)
    expose:
      - "8000"
    networks:
      - backend  # 내부 네트워크만

networks:
  frontend:  # 외부 접근 가능
  backend:   # 내부 전용
    internal: true
```

---

## 옵션 2: AI 서비스도 JWT 검증 (보안 강화)

### 사용 시나리오
- AI 서비스를 직접 외부에 노출하는 경우
- 여러 백엔드 서비스에서 AI 서비스를 호출하는 경우
- 추가 보안이 필요한 경우

### FastAPI JWT 검증 구현

```python
# src/utils/auth.py

from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime
from src.config.settings import settings

security = HTTPBearer()

SECRET_KEY = settings.JWT_SECRET_KEY  # Spring Boot와 동일한 키 사용
ALGORITHM = "HS512"


def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    JWT 토큰 검증

    Spring Boot에서 발급한 JWT 토큰을 검증합니다.
    """
    token = credentials.credentials

    try:
        # 토큰 디코딩
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # 만료 시간 체크
        exp = payload.get("exp")
        if exp and datetime.fromtimestamp(exp) < datetime.now():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )

        # 사용자 ID 추출
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

        return {
            "user_id": user_id,
            "email": payload.get("email"),
            "payload": payload
        }

    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


# 선택적 인증 (토큰이 있으면 검증, 없어도 허용)
def optional_auth(credentials: HTTPAuthorizationCredentials = Security(HTTPBearer(auto_error=False))):
    """선택적 인증"""
    if credentials is None:
        return None
    return verify_jwt_token(credentials)
```

### 보호된 엔드포인트

```python
# src/api/routes.py

from src.utils.auth import verify_jwt_token, optional_auth

@app.post("/api/v1/analyze/match")
async def analyze_match(
    request: MatchRequest,
    user_info = Depends(verify_jwt_token)  # JWT 검증 필수
):
    """
    매치 분석 - JWT 인증 필요

    Args:
        request: 분석 요청
        user_info: JWT에서 추출한 사용자 정보
    """
    print(f"Request from user: {user_info['user_id']}")

    result = match_analyzer.analyze_match(
        match_id=request.match_id,
        summoner_name=request.summoner_name,
        tag_line=request.tag_line,
    )
    return result


# 인증 선택적 엔드포인트 (공개 API)
@app.get("/api/v1/tiers")
async def get_available_tiers(user_info = Depends(optional_auth)):
    """
    티어 목록 조회 - 인증 선택

    로그인한 사용자는 추가 정보 제공
    """
    tiers = gap_analyzer.tier_baseline.get_all_tiers()

    response = {"tiers": tiers}

    if user_info:
        # 로그인 사용자에게는 추가 정보
        response["user_tier"] = "DIAMOND"  # DB에서 조회

    return response
```

### Spring Boot에서 JWT 전달

```java
// AIAnalysisService.java
@Service
public class AIAnalysisService {

    private final WebClient webClient;

    public Mono<MatchAnalysisResult> analyzeMatch(String matchId, String summonerName, String jwtToken) {
        return webClient.post()
            .uri("/api/v1/analyze/match")
            .header("Authorization", "Bearer " + jwtToken)  // JWT 토큰 전달
            .bodyValue(new MatchRequest(matchId, summonerName, "KR1"))
            .retrieve()
            .bodyToMono(MatchAnalysisResult.class);
    }
}

// Controller에서 호출
@RestController
@RequestMapping("/api/analysis")
public class AnalysisController {

    @PostMapping("/match")
    public ResponseEntity<MatchAnalysisResult> analyzeMatch(
        @RequestBody MatchRequest request,
        @RequestHeader("Authorization") String authHeader
    ) {
        String token = authHeader.substring(7); // "Bearer " 제거

        MatchAnalysisResult result = aiAnalysisService.analyzeMatch(
            request.getMatchId(),
            request.getSummonerName(),
            token  // AI 서비스로 JWT 전달
        ).block();

        return ResponseEntity.ok(result);
    }
}
```

---

## 옵션 3: API Key 방식 (간단한 보안)

### Spring Boot와 AI 서비스 간 API Key 공유

```python
# src/utils/auth.py

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from src.config.settings import settings

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

INTERNAL_API_KEY = settings.INTERNAL_API_KEY  # 환경 변수로 관리


async def verify_api_key(api_key: str = Security(api_key_header)):
    """API Key 검증"""
    if api_key != INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return True


# 사용
@app.post("/api/v1/analyze/match")
async def analyze_match(
    request: MatchRequest,
    authorized: bool = Depends(verify_api_key)
):
    """API Key로 보호된 엔드포인트"""
    result = match_analyzer.analyze_match(...)
    return result
```

```java
// Spring Boot에서 API Key 전달
public Mono<MatchAnalysisResult> analyzeMatch(String matchId, String summonerName) {
    return webClient.post()
        .uri("/api/v1/analyze/match")
        .header("X-API-Key", internalApiKey)  // API Key 전달
        .bodyValue(new MatchRequest(matchId, summonerName, "KR1"))
        .retrieve()
        .bodyToMono(MatchAnalysisResult.class);
}
```

---

## 권장 사항

### 개발 단계
- **옵션 1** (인증 없음) 사용
- 빠른 개발 및 테스트
- Docker Compose로 내부 네트워크 격리

### 프로덕션 배포
- **옵션 1** + 네트워크 레벨 보안
- Kubernetes의 경우: NetworkPolicy 사용
- AWS의 경우: VPC Security Group 설정
- AI 서비스는 Private Subnet에 배치

### 멀티 백엔드 환경
- **옵션 2** (JWT) 또는 **옵션 3** (API Key)
- API Gateway 도입 고려

---

## 환경 변수 설정

```bash
# Spring Boot (.env 또는 application.yml)
JWT_SECRET=your-very-secret-key-at-least-256-bits
JWT_EXPIRATION=86400000  # 24시간

# AI Service (.env)
# 옵션 1: 인증 불필요 (비워둠)

# 옵션 2: JWT 사용 시
JWT_SECRET_KEY=your-very-secret-key-at-least-256-bits  # Spring Boot와 동일

# 옵션 3: API Key 사용 시
INTERNAL_API_KEY=your-internal-api-key-for-backend-communication
```

---

## 보안 체크리스트

- [ ] JWT Secret Key는 256bit 이상
- [ ] Secret은 환경 변수로 관리 (GitHub Secrets)
- [ ] HTTPS 사용 (프로덕션)
- [ ] CORS 설정 (허용된 origin만)
- [ ] Rate Limiting 적용
- [ ] API 로그 기록 (감사 목적)
- [ ] 민감한 정보는 로그에 기록 금지
- [ ] 정기적인 Secret 로테이션

---

## 결론

**현재 프로젝트 추천**: **옵션 1 (인증 없음)**

이유:
1. AI 서비스는 내부 서비스로만 동작
2. Spring Boot에서 소셜 로그인으로 사용자 인증 처리
3. 네트워크 레벨에서 AI 서비스 보호
4. 간단하고 유지보수 용이

향후 확장 시:
- 여러 백엔드 서비스가 AI를 호출한다면 → 옵션 2 (JWT)
- AI 서비스를 외부 파트너에게 제공한다면 → 옵션 3 (API Key) + Rate Limiting
