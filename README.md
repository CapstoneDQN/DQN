# Pull Request 협업 가이드

이 문서는 우리 프로젝트에서 Pull Request(PR)를 통해 협업하는 방법을 설명합니다. Git과 GitHub를 처음 사용하는 분들도 쉽게 따라할 수 있도록 자세히 안내합니다.

## 목차
1. [Pull Request란?](#pull-request란)
2. [기본 작업 흐름](#기본-작업-흐름)
3. [세부 단계별 가이드](#세부-단계별-가이드)
4. [코드 리뷰 과정](#코드-리뷰-과정)
5. [자주 발생하는 문제 해결](#자주-발생하는-문제-해결)

## Pull Request란?

Pull Request(PR)는 내가 작업한 코드를 프로젝트의 메인 코드베이스에 병합하기 전에 다른 팀원들에게 검토를 요청하는 방법입니다. 이를 통해:

- 코드 품질을 유지할 수 있습니다 (버그 방지, 코드 스타일 일관성 유지)
- 팀원들과 지식을 공유할 수 있습니다
- 변경사항을 문서화하고 추적할 수 있습니다
- 안전하게 코드를 변경할 수 있습니다

## 기본 작업 흐름

PR을 이용한 기본적인 작업 흐름은 다음과 같습니다:

1. 최신 코드를 받아오기 (`git pull`)
2. 새 브랜치 만들기 (`git checkout -b`)
3. 코드 수정하기
4. 변경사항 커밋하기 (`git add`, `git commit`)
5. 브랜치 올리기 (`git push`) 
6. GitHub에서 PR 생성하기
7. 코드 리뷰 받기
8. 필요시 수정하기
9. PR 병합하기

## 세부 단계별 가이드

### 1. 준비하기: 저장소 복제 (처음 1회만)

이미 저장소를 복제했다면 이 단계는 건너뛰세요.

```bash
# 저장소 복제하기
git clone https://github.com/CapstoneDQN/DQN.git

# 복제한 폴더로 이동
cd 저장소이름
```

### 2. 최신 코드 받아오기

작업을 시작하기 전에 항상 최신 코드를 받아옵니다:

```bash
# main 브랜치로 이동
git checkout main

# 최신 변경사항 가져오기
git pull
```

### 3. 새 브랜치 만들기

각 작업은 별도의 브랜치에서 진행합니다:

```bash
# 새 브랜치 생성 및 전환
git checkout -b 브랜치이름
```

브랜치 이름은 작업 내용을 간결하게 설명해야 합니다:
- `feature/로그인-기능` - 새 기능 개발
- `bugfix/회원가입-오류` - 버그 수정
- `docs/api-문서` - 문서 작업

### 4. 코드 수정하기

필요한 작업을 진행합니다. 코드 편집기를 사용하여 파일을 수정하세요.

### 5. 변경사항 확인 및 커밋하기

변경사항을 확인하고 커밋합니다:

```bash
# 변경된 파일 확인
git status

# 변경사항 자세히 보기
git diff

# 변경사항 스테이징
git add .     # 모든 변경사항 추가
# 또는
git add 파일명  # 특정 파일만 추가

# 커밋하기
git commit -m "작업 내용 설명"
```

커밋 메시지는 명확하고 구체적으로 작성하세요:
- ✅ 좋은 예: "로그인 페이지 UI 구현"
- ❌ 나쁜 예: "업데이트", "수정"

### 6. 브랜치 올리기

작업한 브랜치를 GitHub에 올립니다:

```bash
git push origin 브랜치이름
```

처음 올리는 경우 다음과 같은 메시지가 나올 수 있습니다:
```
fatal: The current branch 브랜치이름 has no upstream branch.
To push the current branch and set the remote as upstream, use

    git push --set-upstream origin 브랜치이름
```

이 경우 제안된 명령어를 그대로 복사하여 실행하세요:
```bash
git push --set-upstream origin 브랜치이름
```

### 7. Pull Request 생성하기

1. GitHub 웹사이트에서 저장소 페이지로 이동합니다.
2. 방금 올린 브랜치에 대한 알림이 표시됩니다. "Compare & pull request" 버튼을 클릭합니다.
   - 알림이 보이지 않는 경우, "Pull requests" 탭으로 이동한 후 "New pull request" 버튼을 클릭합니다.
3. PR 양식을 작성합니다:
   - 제목: 작업 내용을 간결하게 설명 (예: "로그인 기능 구현")
   - 설명: 변경사항에 대한 자세한 설명, 관련 이슈 번호, 테스트 방법 등
4. "Create pull request" 버튼을 클릭합니다.

### 작성 예시

```
제목: 로그인 페이지 UI 구현

설명:
- 사용자 로그인 페이지의 기본 UI를 구현했습니다
- 아이디/비밀번호 입력 필드 및 로그인 버튼 추가
- 기본적인 입력값 검증 기능 추가

관련 이슈: #42

테스트 방법:
1. 웹 페이지 접속
2. 로그인 페이지로 이동
3. 아이디/비밀번호 입력 후 로그인 버튼 동작 확인
```

## 코드 리뷰 과정

PR을 생성한 후에는 다음과 같은 과정이 진행됩니다:

1. **리뷰 요청하기**: PR 페이지에서 우측 사이드바의 "Reviewers"에서 리뷰어를 지정할 수 있습니다.

2. **리뷰 대기하기**: 팀원들이 코드를 검토하고 피드백을 제공합니다.

3. **피드백 반영하기**: 리뷰어가 변경을 요청한 경우:
   ```bash
   # 같은 브랜치에서 코드 수정
   git add .
   git commit -m "리뷰 피드백 반영"
   git push origin 브랜치이름
   ```
   - 추가 커밋은 자동으로 PR에 반영됩니다.

4. **승인 받기**: 모든 리뷰어가 승인하면 PR을 병합할 수 있습니다.

5. **병합하기**: GitHub 웹사이트의 PR 페이지에서 "Merge pull request" 버튼을 클릭합니다.

## 자주 발생하는 문제 해결

### 1. 충돌(Conflict) 발생 시

PR을 생성했는데 충돌이 발생한 경우:

```bash
# main 브랜치의 최신 변경사항 가져오기
git checkout main
git pull

# 작업 브랜치로 돌아가기
git checkout 브랜치이름

# main의 변경사항을 현재 브랜치에 병합
git merge main

# 충돌 해결하기 (코드 편집기에서 충돌 표시된 부분 수정)

# 충돌 해결 후 커밋
git add .
git commit -m "충돌 해결"

# 다시 푸시
git push origin 브랜치이름
```

### 2. 작업 중 main 브랜치가 변경된 경우

작업 도중 main 브랜치에 새로운 변경사항이 병합된 경우, 최신 변경사항을 작업 브랜치에 반영해야 합니다:

```bash
# main 브랜치의 최신 변경사항 가져오기
git checkout main
git pull

# 작업 브랜치로 돌아가기
git checkout 브랜치이름

# main의 변경사항을 현재 브랜치에 리베이스
git rebase main

# 충돌이 있다면 해결한 후
git add .
git rebase --continue

# 강제 푸시 (이미 PR을 생성한 경우에만)
git push --force origin 브랜치이름
```

### 3. 실수로 main 브랜치에서 작업한 경우

```bash
# 변경사항을 스테이징하지 않은 경우
git stash
git checkout -b 새브랜치이름
git stash pop

# 이미 커밋한 경우
git checkout -b 새브랜치이름
git checkout main
git reset --hard origin/main
```

## 추가 팁

- **작은 단위로 PR 생성하기**: 큰 변경사항은 여러 개의 작은 PR로 나누는 것이 리뷰하기 쉽습니다.
- **PR 설명 잘 작성하기**: 변경사항이 무엇인지, 왜 필요한지 명확하게 설명하면 리뷰가 빠르게 진행됩니다.
- **정기적으로 커밋하기**: 작은 변경사항마다 커밋하면 작업 내용을 추적하기 쉽습니다.
- **VSCode에서 GitHub 확장 프로그램 사용하기**: VSCode의 GitHub Pull Requests 확장을 설치하면 에디터 내에서 직접 PR을 관리할 수 있습니다.

언제든지 질문이나 도움이 필요하면 팀원에게 문의하세요! 모두가 처음에는 초보자였습니다. 😊
