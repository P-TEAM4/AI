# Spring Boot 연동 가이드

이 문서는 Spring Boot 백엔드 서버와 AI 서비스를 연동하는 방법을 설명합니다.

## 아키텍처

```
[Spring Boot Backend] <--> [FastAPI AI Service] <--> [Riot Games API]
        |                           |
        v                           v
   [Database]                [ML Models]
```

## API 엔드포인트

### Base URL
- 개발 환경: `http://localhost:8000`
- 프로덕션: `http://your-domain.com/api`

### 인증
현재 버전은 인증이 구현되지 않았습니다. 프로덕션 배포 시 API Key 또는 JWT 토큰 기반 인증을 추가해야 합니다.

## Spring Boot에서 호출하는 방법

### 1. RestTemplate을 사용한 예제

```java
@Service
public class AIAnalysisService {

    private final RestTemplate restTemplate;

    @Value("${ai.service.url}")
    private String aiServiceUrl;

    public AIAnalysisService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public MatchAnalysisResult analyzeMatch(String matchId, String summonerName, String tagLine) {
        String url = aiServiceUrl + "/api/v1/analyze/match";

        MatchRequest request = new MatchRequest(matchId, summonerName, tagLine);

        ResponseEntity<MatchAnalysisResult> response = restTemplate.postForEntity(
            url,
            request,
            MatchAnalysisResult.class
        );

        return response.getBody();
    }

    public ProfileAnalysisResult analyzeProfile(String summonerName, String tagLine, int recentGames) {
        String url = aiServiceUrl + "/api/v1/analyze/profile";

        ProfileRequest request = new ProfileRequest(summonerName, tagLine, recentGames);

        ResponseEntity<ProfileAnalysisResult> response = restTemplate.postForEntity(
            url,
            request,
            ProfileAnalysisResult.class
        );

        return response.getBody();
    }

    public GapAnalysisResult analyzeGap(PlayerStats playerStats, String tier, String division) {
        String url = aiServiceUrl + "/api/v1/analyze/gap";

        GapAnalysisRequest request = new GapAnalysisRequest(playerStats, tier, division);

        ResponseEntity<GapAnalysisResult> response = restTemplate.postForEntity(
            url,
            request,
            GapAnalysisResult.class
        );

        return response.getBody();
    }
}
```

### 2. WebClient를 사용한 예제 (권장)

```java
@Service
public class AIAnalysisService {

    private final WebClient webClient;

    public AIAnalysisService(@Value("${ai.service.url}") String aiServiceUrl) {
        this.webClient = WebClient.builder()
            .baseUrl(aiServiceUrl)
            .build();
    }

    public Mono<MatchAnalysisResult> analyzeMatch(String matchId, String summonerName, String tagLine) {
        MatchRequest request = new MatchRequest(matchId, summonerName, tagLine);

        return webClient.post()
            .uri("/api/v1/analyze/match")
            .bodyValue(request)
            .retrieve()
            .bodyToMono(MatchAnalysisResult.class);
    }

    public Mono<ProfileAnalysisResult> analyzeProfile(String summonerName, String tagLine, int recentGames) {
        ProfileRequest request = new ProfileRequest(summonerName, tagLine, recentGames);

        return webClient.post()
            .uri("/api/v1/analyze/profile")
            .bodyValue(request)
            .retrieve()
            .bodyToMono(ProfileAnalysisResult.class);
    }
}
```

### 3. DTO 클래스 예제

```java
@Data
@NoArgsConstructor
@AllArgsConstructor
public class MatchRequest {
    private String matchId;
    private String summonerName;
    private String tagLine;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ProfileRequest {
    private String summonerName;
    private String tagLine;
    private Integer recentGames = 20;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
public class GapAnalysisRequest {
    private PlayerStats playerStats;
    private String tier;
    private String division;
}

@Data
public class PlayerStats {
    private Integer kills;
    private Integer deaths;
    private Integer assists;
    private Double kda;
    private Integer cs;
    private Double csPerMin;
    private Integer gold;
    private Integer visionScore;
    private Integer damageDealt;
    private Double damageShare;
    private String championName;
    private String position;
}

@Data
public class GapAnalysisResult {
    private String tier;
    private Map<String, Double> playerAvg;
    private Map<String, Double> tierAvg;
    private Map<String, Double> gaps;
    private Map<String, Double> normalizedGaps;
    private Double overallScore;
    private List<String> strengths;
    private List<String> weaknesses;
    private List<String> recommendations;
}
```

### 4. Controller 예제

```java
@RestController
@RequestMapping("/api/analysis")
@RequiredArgsConstructor
public class AnalysisController {

    private final AIAnalysisService aiAnalysisService;

    @PostMapping("/match")
    public ResponseEntity<MatchAnalysisResult> analyzeMatch(@RequestBody MatchRequest request) {
        MatchAnalysisResult result = aiAnalysisService.analyzeMatch(
            request.getMatchId(),
            request.getSummonerName(),
            request.getTagLine()
        );

        return ResponseEntity.ok(result);
    }

    @PostMapping("/profile")
    public ResponseEntity<ProfileAnalysisResult> analyzeProfile(@RequestBody ProfileRequest request) {
        ProfileAnalysisResult result = aiAnalysisService.analyzeProfile(
            request.getSummonerName(),
            request.getTagLine(),
            request.getRecentGames()
        );

        return ResponseEntity.ok(result);
    }

    @PostMapping("/gap")
    public ResponseEntity<GapAnalysisResult> analyzeGap(@RequestBody GapAnalysisRequest request) {
        GapAnalysisResult result = aiAnalysisService.analyzeGap(
            request.getPlayerStats(),
            request.getTier(),
            request.getDivision()
        );

        return ResponseEntity.ok(result);
    }
}
```

### 5. application.yml 설정

```yaml
ai:
  service:
    url: http://localhost:8000
    timeout: 30000  # 30 seconds

spring:
  application:
    name: lol-highlight-backend
```

## 에러 처리

AI 서비스에서 반환되는 HTTP 상태 코드:

- `200 OK`: 성공
- `404 Not Found`: 매치 또는 플레이어를 찾을 수 없음
- `500 Internal Server Error`: 서버 에러

Spring Boot에서 에러 처리 예제:

```java
@ControllerAdvice
public class AIServiceExceptionHandler {

    @ExceptionHandler(WebClientResponseException.class)
    public ResponseEntity<ErrorResponse> handleWebClientException(WebClientResponseException ex) {
        ErrorResponse error = new ErrorResponse(
            ex.getStatusCode().value(),
            ex.getMessage()
        );

        return ResponseEntity.status(ex.getStatusCode()).body(error);
    }
}
```

## 비동기 처리

분석 작업은 시간이 걸릴 수 있으므로 비동기 처리를 권장합니다:

```java
@Service
public class AsyncAIAnalysisService {

    private final AIAnalysisService aiAnalysisService;

    @Async
    public CompletableFuture<MatchAnalysisResult> analyzeMatchAsync(
        String matchId,
        String summonerName,
        String tagLine
    ) {
        MatchAnalysisResult result = aiAnalysisService.analyzeMatch(matchId, summonerName, tagLine);
        return CompletableFuture.completedFuture(result);
    }
}
```

## 캐싱

분석 결과를 캐싱하여 성능을 향상시킬 수 있습니다:

```java
@Service
public class CachedAIAnalysisService {

    private final AIAnalysisService aiAnalysisService;

    @Cacheable(value = "matchAnalysis", key = "#matchId")
    public MatchAnalysisResult analyzeMatch(String matchId, String summonerName, String tagLine) {
        return aiAnalysisService.analyzeMatch(matchId, summonerName, tagLine);
    }
}
```

## 데이터베이스 저장

분석 결과를 데이터베이스에 저장하는 예제:

```java
@Entity
@Table(name = "match_analysis")
public class MatchAnalysisEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true)
    private String matchId;

    private String summonerName;

    @Column(columnDefinition = "json")
    private String analysisResult;

    private LocalDateTime analyzedAt;

    // getters, setters
}

@Service
@Transactional
public class AnalysisStorageService {

    private final MatchAnalysisRepository repository;
    private final AIAnalysisService aiAnalysisService;
    private final ObjectMapper objectMapper;

    public MatchAnalysisResult getOrAnalyzeMatch(String matchId, String summonerName, String tagLine) {
        Optional<MatchAnalysisEntity> existing = repository.findByMatchId(matchId);

        if (existing.isPresent()) {
            // Return cached result from database
            return objectMapper.readValue(existing.get().getAnalysisResult(), MatchAnalysisResult.class);
        }

        // Analyze and save
        MatchAnalysisResult result = aiAnalysisService.analyzeMatch(matchId, summonerName, tagLine);

        MatchAnalysisEntity entity = new MatchAnalysisEntity();
        entity.setMatchId(matchId);
        entity.setSummonerName(summonerName);
        entity.setAnalysisResult(objectMapper.writeValueAsString(result));
        entity.setAnalyzedAt(LocalDateTime.now());

        repository.save(entity);

        return result;
    }
}
```

## 테스트

Spring Boot에서 AI 서비스를 테스트하는 방법:

```java
@SpringBootTest
@AutoConfigureMockMvc
class AnalysisControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private AIAnalysisService aiAnalysisService;

    @Test
    void testAnalyzeMatch() throws Exception {
        MatchRequest request = new MatchRequest("KR_123", "Player", "KR1");

        mockMvc.perform(post("/api/analysis/match")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
            .andExpect(status().isOk());
    }
}
```

## 배포

### Docker Compose를 사용한 배포

```yaml
version: '3.8'

services:
  ai-service:
    build: ./AI
    ports:
      - "8000:8000"
    environment:
      - RIOT_API_KEY=${RIOT_API_KEY}
    networks:
      - app-network

  spring-boot:
    build: ./backend
    ports:
      - "8080:8080"
    environment:
      - AI_SERVICE_URL=http://ai-service:8000
    depends_on:
      - ai-service
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
```

## 보안 고려사항

1. **API Key 보호**: Riot API Key는 환경 변수 또는 GitHub Secrets로 관리
2. **CORS 설정**: 프로덕션에서는 허용된 origin만 설정
3. **Rate Limiting**: Riot API 호출 제한 준수
4. **입력 검증**: 모든 사용자 입력 검증
5. **에러 메시지**: 민감한 정보가 노출되지 않도록 주의
