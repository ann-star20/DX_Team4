# 공개 범위 및 자격증명 점검

점검일: 2026-11-11. 로컬 작업 트리와 로컬에 존재하는 Git refs 기준입니다. 원격 GitHub 공개 상태·키 유효성은 확인하거나 변경하지 않았습니다.

## 결론: 이력 정리 완료, 자격증명 교체 확인 전 공개 보류

.env.example의 SUPABASE_DB_URL에 예시가 아닌 형태의 비밀번호와 프로젝트 주소가 포함되어 있었습니다. 값을 출력하거나 실제 접속을 시도하지 않았습니다. SUPABASE_KEY는 예시값 형태였습니다.

- 현재 작업 파일: DB URL·프로젝트 URL을 placeholder로 교체.
- 이력: aefe322ecde2, 0c7696cddd10의 .env.example에서 동일 DB URL 확인.
- 로컬에 알려진 origin/main 및 다수 feature/backup refs에도 같은 문자열 존재.
- 파일 수정 커밋은 과거 비밀값을 제거하지 않습니다.
- 비밀번호가 현재 유효한지·이미 교체됐는지는 미확인입니다.

## 공개에 포함할 자료

| 범위 | 내용 |
| --- | --- |
| app/, src/ | 애플리케이션·모델 로더·전처리 소스 |
| static/ | 서비스 UI. 데이터·자산 권리 별도 검토 필요 |
| scripts/ | 유지보수와 공개 전 점검 코드 |
| README.md, requirements*.txt | 실행·재현 조건 |
| docs/images/ | 검토된 공개용 다이어그램 |
| .env.example | 실제 비밀값 없는 예시만 허용 |

## 공개 제외: 로컬 보존

portfolio.html(오타 portpolio.html 포함), architecture.html, tech.html, notion-daily-system.md, outputs-all/, 모델_스크립트/, 코드기반_분석_MD/는 내부·생성 자료로 제외합니다. 실제 서비스 UI인 static/과 혼동하지 않습니다.

.env, 서비스 계정 JSON, 키 파일, 자격증명 디렉터리, 모델·데이터·녹음·DB 덤프도 제외합니다. ignore 규칙은 기존 커밋을 삭제하지 않습니다. 포트폴리오 HTML 역시 과거 커밋까지 비공개로 만들려면 별도 이력 정리가 필요합니다.

## 저장소 소유자가 마쳐야 할 조치

1. Supabase Dashboard에서 노출된 DB 비밀번호를 교체하고 해당 DB를 사용하는 배포·로컬 설정을 갱신.
2. 백엔드 권한 키도 별도로 노출된 적이 있다면 그 키를 회전. 프로젝트 URL 자체는 비밀번호가 아니지만 공개 예시에서는 제거.
3. 모든 원격 브랜치·태그와 PR 참조에 대해 비밀값 이력 제거 범위 확정. 공유 저장소 이력 재작성·강제 push는 협업자와 조율 후 수행.
4. 새 clone에서 전체 이력 검사. 기존 clone·fork·PR 캐시의 잔존도 고려.
5. 서비스 이미지·폰트·음원 및 팀 기여 코드의 공개 권한 확인.
6. 변경 커밋·push와 원격 상태 재점검 후 public 전환.

후속 승인으로 커밋·push와 15개 원격 브랜치 이력 정리를 실행했습니다. 계정 비밀번호와 저장소 가시성은 변경하지 않았습니다.

원래 저장소 이력을 공개할 필요가 없다면, 검토된 파일만 새 디렉터리·새 저장소로 옮기는 방법도 있습니다. 기존 .git을 복사하거나 기존 remote를 연결한 채 전체 branch를 push하면 안 됩니다. 이 경우에도 노출된 자격증명의 회전은 필요합니다.

## 서비스 운영과 소스 공개의 구분

현재 API의 user_id 기반 검색 필터는 접근 권한 검증이 아닙니다. 실제 사용자 데이터를 연결한 공개 서버 운영 전에는 인증·인가, RLS, CORS 허용 범위, 업로드 제한, 비용·호출 제한을 검증해야 합니다. 소스 공개만으로 이러한 운영 검증이 끝나지는 않습니다.

## 검사 범위

scripts/check_public_release.py는 현재 추적 파일과 선택적으로 전체 로컬 refs의 blob에서 비밀번호 포함 PostgreSQL URL, JWT, Google API 키, private key, Supabase secret 패턴을 검사합니다. 값은 출력하지 않습니다. 알려지지 않은 원격 refs, GitHub PR 캐시·fork, 미등록 형식의 비밀값·개인정보는 탐지 보장 범위 밖입니다.

## 공식 지침

- [Supabase DB 비밀번호 재설정](https://supabase.com/kb/troubleshooting/how-do-i-reset-my-supabase-database-password-oTs5sB)
- [GitHub 민감정보 이력 제거](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)

## 이번 작업 검증

- 내부 HTML·Notion 초안 4개는 Git index에서 제외, 로컬 보존 확인.
- 전체 로컬 refs의 blob 212개 패턴 검사: 과거 .env.example blob 2개에서 DB 비밀번호 패턴 발견.
- 현재 작업 파일의 값 교체는 완료했으나 과거 이력은 그대로이므로 공개 준비 완료 상태가 아닙니다.
- 실행 의존성의 누락 직접 import를 보완했지만, 새 환경 설치·실제 GCP/DB/모델 실행은 수행하지 않았습니다.

## 2026-11-11 후속 이력 정리 결과

- 공개 준비 변경사항을 main에 커밋·push한 뒤 별도 복제본에서 이력을 정리했습니다.
- 15개 원격 브랜치에서 과거 .env.example, portfolio.html, architecture.html, tech.html, notion-daily-system.md를 제거했습니다.
- main에는 비밀값 없는 .env.example을 새 커밋으로 복원했습니다(1becb8f).
- atomic push와 브랜치별 force-with-lease로 예상하지 않은 원격 변경 덮어쓰기를 방지했습니다.
- 정리된 이력의 blob 218개 검사에서 비밀정보 패턴이 발견되지 않았습니다. 이는 기존 clone·fork·PR 캐시 제거를 보장하지 않습니다.
- 로컬 작업 파일과 비공개 HTML은 보존했습니다. 협업자는 새 clone을 사용하고 구 이력을 merge/push하지 않아야 합니다.
- DB 비밀번호 교체 여부는 미확인이므로 public 전환은 계속 보류합니다.
