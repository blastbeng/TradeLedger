import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from src.web.app import app, set_engine


@pytest.fixture
def client():
    with patch("src.web.app.get_redis_client") as mock_redis, \
         patch("src.web.app.check_redis_connection", return_value=True), \
         patch("src.web.app.is_redis_available", return_value=True), \
         patch("src.web.app.settings") as mock_settings, \
         patch("src.web.app.run_in_threadpool", new=AsyncMock(side_effect=lambda f, *args, **kwargs: f(*args, **kwargs))):
        
        # Disable auth and CSRF for testing
        mock_settings.WEB_USERNAME = ""
        mock_settings.WEB_PASSWORD = ""
        mock_settings.WEB_RATE_LIMIT_REQUESTS = 1000
        mock_settings.WEB_RATE_LIMIT_WINDOW = 60
        
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.zcard.return_value = 0
        mock_redis_client.get.return_value = None
        mock_redis_client.lrange.return_value = []
        
        # Mock the engine
        mock_engine = MagicMock()
        mock_engine._is_market_open = AsyncMock(return_value=True)
        mock_engine.current_symbols = []
        mock_engine.positions = {}
        mock_engine.queued_orders = []
        mock_engine.trade_history = []
        mock_engine.trader.fetch_balance.return_value = {"EUR": 10000.0}
        mock_engine.get_pause_status = AsyncMock(return_value={})
        mock_engine.get_performance_summary = AsyncMock(return_value={"rows": [], "total": {}})
        mock_engine.event_bus.request = AsyncMock(return_value={})
        mock_engine._market_data_manager._get_quotes_async = AsyncMock(return_value={})
        mock_engine._get_stock_name = AsyncMock(return_value="Test")
        mock_engine._format_symbol_display.return_value = "TEST"
        
        set_engine(mock_engine)
        
        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    with patch("src.web.app.check_llm_health", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "mind": {"status": "connected"},
            "actuator": {"status": "connected"},
            "weak": {"status": "connected"}
        }
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_status_endpoint(client):
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert "current_symbols" in data
    assert "positions" in data
    assert "balances" in data


def test_config_endpoint(client):
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert "trading_mode" in data
    assert "base_currency" in data


def test_pause_resume(client):
    response = client.post("/api/v1/pause")
    assert response.status_code == 200
    assert response.json()["status"] == "paused"
    
    response = client.post("/api/v1/resume")
    assert response.status_code == 200
    assert response.json()["status"] == "resumed"


def test_force_reeval(client):
    response = client.post("/api/v1/force-reeval")
    assert response.status_code == 200
    assert response.json()["status"] == "Forced re-evaluation triggered"
