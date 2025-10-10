#!/bin/bash
# Qwen2.5-Omni AWQ Service Test Script

echo "================================================"
echo "Qwen2.5-Omni AWQ Service Test"
echo "================================================"
echo ""

# Check if service is running
echo "1. Checking if AWQ service is running..."
if docker ps | grep -q "qwen2.5-omni-http-awq"; then
    echo "✓ AWQ service is running"
else
    echo "✗ AWQ service is not running"
    echo ""
    echo "To start the service, run:"
    echo "  docker-compose --profile awq up -d"
    exit 1
fi

echo ""

# Wait for service to be ready
echo "2. Waiting for service to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo "✓ Service is ready!"
        break
    fi
    echo "   Waiting... ($((RETRY_COUNT + 1))/$MAX_RETRIES)"
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "✗ Service failed to start within timeout"
    echo ""
    echo "Check logs with:"
    echo "  docker-compose logs qwen2.5-omni-http-awq"
    exit 1
fi

echo ""

# Health check
echo "3. Health check..."
HEALTH=$(curl -s http://localhost:5000/health)
echo "$HEALTH" | python3 -m json.tool
echo ""

# Test transcription (if test audio file exists)
if [ -f "test_audio.wav" ]; then
    echo "4. Testing transcription..."
    echo "   Uploading test_audio.wav..."

    RESPONSE=$(curl -s -X POST \
        -F "file=@test_audio.wav" \
        http://localhost:5000/transcribe/json)

    echo "$RESPONSE" | python3 -m json.tool
    echo ""
    echo "✓ Transcription test complete!"
else
    echo "4. Skipping transcription test (test_audio.wav not found)"
    echo ""
    echo "To test transcription, place an audio file named 'test_audio.wav' in this directory and run again."
fi

echo ""
echo "================================================"
echo "Available Endpoints:"
echo "================================================"
echo "POST /transcribe         - Returns text file"
echo "POST /transcribe/json    - Returns JSON"
echo "POST /transcribe/srt     - Returns SRT subtitle"
echo "POST /transcribe/async   - Async transcription"
echo "GET  /status/<job_id>    - Check async job status"
echo "GET  /health             - Health check"
echo ""
echo "Example curl commands:"
echo "  curl -X POST -F 'file=@audio.mp3' http://localhost:5000/transcribe/json"
echo "  curl -X POST -F 'file=@audio.mp3' http://localhost:5000/transcribe/srt -o output.srt"
echo "  curl http://localhost:5000/health"
echo ""
