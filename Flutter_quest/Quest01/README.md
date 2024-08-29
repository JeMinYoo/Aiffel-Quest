# AIFFEL Campus Code Peer Review Templete
- 코더 : 유제민
- 리뷰어 : 김재이


# PRT(Peer Review Template)
[O]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
- 문제에서 요구하는 기능이 정상적으로 작동합니다.
  ```
  void main() {
  PomodoroTimer pomodoroTimer = PomodoroTimer();
  pomodoroTimer.start();
}
  ```

[O]  **2. 핵심적이거나 복잡하고 이해하기 어려운 부분에 작성된 설명을 보고 해당 코드가 잘 이해되었나요?**
- 해당 코드 블럭에 doc string/annotation/markdown이 달려 있는지 확인
- 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석이 꼼꼼히 작성되어 있어 이해하기 쉬웠습니다.

```
    timer = Timer.periodic(Duration(seconds: 1), (Timer t){
      if (currentTime > 0) { // 설정된 시간이 안 끝났다면
        currentTime--; // 남은 시간을 1씩 빼고
        print('Flutter : ${format(currentTime)}'); // 남은 시간 출력
      }
      else { // 설정된 시간이 끝났다면
        if (isWorking) { // 현재 업무 시간이였다면
          cycleCount++; // 회차를 더하고
          print('$cycleCount 회차 업무 시간 종료. 휴식 시작');
          currentTime = cycleCount % 4 == 0 ? longRest : shortRest; // 4회차라면 15분 휴식 아니라면 5분 휴식
        }
        else{ // 현재 휴식 시간이였다면
          print('휴식 시간 종료. ${cycleCount+1} 회차 업무 시작');
          currentTime = workTime; // 25분을 부여
        }
        isWorking = !isWorking; // 업무 또는 휴식 시간 변경
```

[O]  **3. 에러가 난 부분을 디버깅하여 “문제를 해결한 기록”을 남겼나요? 또는 “새로운 시도 및 추가 실험”을 해봤나요?**
- 문제 원인 및 해결 과정을 잘 기록하였는지 확인
- 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 실험이 기록되어 있는지 확인
    - 어떤 시도를 하며 프로그램을 작성하셨는지 알 수 있습니다.

```
  // 처음에 workTime이나 Rest를 초 단위로 만들어 놔서 1500초...300초 이렇게 나왔다.
  // 이를 바꾸기 위해 format 작성
```
     
[O]  **4. 회고를 잘 작성했나요?**
- 프로젝트 결과물에 대해 배운점과 아쉬운점, 느낀점 등이 상세히 기록 되어 있나요?
    - 회고가 잘 작성되어 있습니다.

```
```


[O]  **5. 코드가 간결하고 효율적인가요?**
- 파이썬 스타일 가이드 (PEP8)를 준수하였는지 확인
- 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 모듈화(함수화) 했는지
    - 포모도로 클래스의 주요 메서드도 간결했지만, 출력 내용을 따로 포맷팅하신 부분과 선언된 변수를 보아도 코드가 유추되도록 작성해주셨습니다.

```
  static const int workTime = 5; // 25 분 업무 시간 / 동작을 보기 위해 시간 단축
  static const int shortRest = 5; // 5 분 4회차를 제외한 휴식 시간 / 동작을 보기 위해 시간 단축
  static const int longRest = 10; // 15 분 4회차 휴식 시간 / 동작을 보기 위해 시간 단축
  int currentTime = workTime; // currentTime은 실시간으로 줄어들면서 타이머 역할을 함
  Timer? timer; // null 값이 들어갈 수 있도록 ? 추가
  bool isWorking = true; // 현재 휴식 시간인지 업무 시간인지 체크 true 업무, false 휴식
  int cycleCount = 0; // 반복 회차
```

```
  String format(int seconds){ // 초 단위를 분 단위로 변환
    var duration = Duration(seconds: seconds);
    return duration.toString().split('.').first.substring(2, 7);
  }

```
