FROM python:3.9

# 디렉터리 지정
RUN mkdir /app
WORKDIR /app

# 현재 디렉터리에 있는 파일 복사
COPY ./ ./

# 모듈 설치
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 포트 열기
EXPOSE 8080

# 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

