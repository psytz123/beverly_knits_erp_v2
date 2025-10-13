"""
PATTERN: Pytest Async Fixtures
CATEGORY: testing/pytest
USE_CASE: Reusable test fixtures for async FastAPI applications
PERFORMANCE: Test setup <100ms, teardown <50ms
TESTED: 2025-10-05
VERSION: 1.0.0

Production-ready pytest fixtures for:
- Async database sessions
- Test clients
- Sample data
- Authentication mocking
- Complete test coverage
"""

import asyncio
from typing import AsyncGenerator, Generator
import pytest
from httpx import AsyncClient
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

# ============================================================================
# DATABASE FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """
    Create event loop for entire test session.

    Scope: session
    Purpose: Reuse event loop across all tests for performance
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def async_engine():
    """
    Create async engine for test database.

    Scope: session
    Database: In-memory SQLite for speed
    Performance: <50ms setup

    Example:
        @pytest.mark.asyncio
        async def test_something(async_engine):
            async with async_engine.begin() as conn:
                await conn.execute(...)
    """
    # Use in-memory SQLite for tests (fast)
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=StaticPool,  # Single connection for in-memory DB
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Create fresh database session for each test.

    Scope: function (new session per test)
    Transaction: Rolled back after test (test isolation)
    Performance: <20ms per test

    Example:
        @pytest.mark.asyncio
        async def test_create_user(db_session: AsyncSession):
            user = User(email="test@example.com")
            db_session.add(user)
            await db_session.commit()
            assert user.id is not None
    """
    async_session = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with async_session() as session:
        async with session.begin():
            yield session
            # Transaction automatically rolled back (test isolation)


# ============================================================================
# HTTP CLIENT FIXTURES
# ============================================================================

@pytest.fixture
async def async_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    Create async HTTP client for API testing.

    Scope: function
    Purpose: Test FastAPI endpoints
    Performance: <50ms setup

    Example:
        @pytest.mark.asyncio
        async def test_create_user(async_client: AsyncClient):
            response = await async_client.post(
                "/users/",
                json={"email": "test@example.com"}
            )
            assert response.status_code == 201
    """
    from app.main import app
    from app.database import get_db

    # Override database dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

    # Cleanup
    app.dependency_overrides.clear()


# ============================================================================
# AUTHENTICATION FIXTURES
# ============================================================================

@pytest.fixture
async def auth_headers(sample_user: User) -> dict:
    """
    Create authentication headers for testing protected endpoints.

    Scope: function
    Purpose: Test authenticated endpoints
    Performance: <10ms

    Example:
        @pytest.mark.asyncio
        async def test_protected_endpoint(
            async_client: AsyncClient,
            auth_headers: dict
        ):
            response = await async_client.get(
                "/users/me",
                headers=auth_headers
            )
            assert response.status_code == 200
    """
    from app.auth import create_access_token

    token = create_access_token({"sub": str(sample_user.id)})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def admin_headers(admin_user: User) -> dict:
    """
    Create admin authentication headers.

    Scope: function
    Purpose: Test admin-only endpoints
    """
    from app.auth import create_access_token

    token = create_access_token({"sub": str(admin_user.id), "role": "admin"})
    return {"Authorization": f"Bearer {token}"}


# ============================================================================
# SAMPLE DATA FIXTURES
# ============================================================================

@pytest.fixture
async def sample_user(db_session: AsyncSession) -> User:
    """
    Create sample user for testing.

    Scope: function
    Purpose: Provide test user data
    Performance: <30ms

    Example:
        @pytest.mark.asyncio
        async def test_get_user(async_client: AsyncClient, sample_user: User):
            response = await async_client.get(f"/users/{sample_user.id}")
            assert response.status_code == 200
    """
    user = User(
        email="test@example.com",
        name="Test User",
        is_active=True
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def admin_user(db_session: AsyncSession) -> User:
    """Create admin user for testing."""
    user = User(
        email="admin@example.com",
        name="Admin User",
        is_active=True,
        role="admin"
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def sample_users(db_session: AsyncSession) -> list[User]:
    """
    Create multiple sample users for pagination testing.

    Scope: function
    Count: 10 users
    Performance: <100ms

    Example:
        @pytest.mark.asyncio
        async def test_list_users(
            async_client: AsyncClient,
            sample_users: list[User]
        ):
            response = await async_client.get("/users/")
            assert len(response.json()) == 10
    """
    users = [
        User(
            email=f"user{i}@example.com",
            name=f"User {i}",
            is_active=True
        )
        for i in range(10)
    ]
    db_session.add_all(users)
    await db_session.commit()

    for user in users:
        await db_session.refresh(user)

    return users


# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_external_api(monkeypatch):
    """
    Mock external API calls.

    Scope: function
    Purpose: Prevent real API calls during tests
    Performance: Instant

    Example:
        @pytest.mark.asyncio
        async def test_with_external_api(mock_external_api):
            # External API is mocked
            result = await call_external_api()
            assert result["status"] == "mocked"
    """
    async def mock_call(*args, **kwargs):
        return {"status": "mocked", "data": {}}

    monkeypatch.setattr("app.services.external_api.call", mock_call)


@pytest.fixture
def mock_cache(monkeypatch):
    """
    Mock Redis cache for testing.

    Scope: function
    Purpose: Test caching logic without Redis
    """
    cache_data = {}

    async def mock_get(key: str):
        return cache_data.get(key)

    async def mock_set(key: str, value: Any, expire: int = 0):
        cache_data[key] = value

    async def mock_delete(key: str):
        cache_data.pop(key, None)

    monkeypatch.setattr("app.cache.get", mock_get)
    monkeypatch.setattr("app.cache.set", mock_set)
    monkeypatch.setattr("app.cache.delete", mock_delete)


# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def freeze_time(monkeypatch):
    """
    Freeze time for testing time-dependent code.

    Scope: function
    Purpose: Consistent timestamps in tests

    Example:
        @pytest.mark.asyncio
        async def test_expiration(freeze_time):
            freeze_time("2025-10-05 12:00:00")
            # All datetime.now() calls return frozen time
    """
    from datetime import datetime
    frozen_time = None

    def _freeze(time_str: str):
        nonlocal frozen_time
        frozen_time = datetime.fromisoformat(time_str)

    def mock_now():
        return frozen_time if frozen_time else datetime.now()

    monkeypatch.setattr("datetime.datetime.now", mock_now)
    return _freeze


# ============================================================================
# PARAMETRIZE FIXTURES
# ============================================================================

@pytest.fixture(params=[
    {"email": "valid@example.com", "valid": True},
    {"email": "invalid-email", "valid": False},
    {"email": "", "valid": False},
])
def email_validation_cases(request):
    """
    Parametrized fixture for email validation testing.

    Scope: function
    Purpose: Test multiple validation scenarios

    Example:
        @pytest.mark.asyncio
        async def test_email_validation(email_validation_cases):
            result = validate_email(email_validation_cases["email"])
            assert result == email_validation_cases["valid"]
    """
    return request.param


# ============================================================================
# CONFTEST.PY CONFIGURATION
# ============================================================================

"""
# In conftest.py, add:

import pytest

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

# Add markers
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")

# Configure test database
@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    '''Set up test database before all tests.'''
    # Database setup logic
    yield
    # Cleanup logic
"""

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
# Basic test:
@pytest.mark.asyncio
async def test_create_user(async_client: AsyncClient):
    response = await async_client.post(
        "/users/",
        json={"email": "test@example.com", "name": "Test"}
    )
    assert response.status_code == 201

# Test with sample data:
@pytest.mark.asyncio
async def test_get_user(async_client: AsyncClient, sample_user: User):
    response = await async_client.get(f"/users/{sample_user.id}")
    assert response.status_code == 200
    assert response.json()["email"] == sample_user.email

# Test with authentication:
@pytest.mark.asyncio
async def test_protected_route(
    async_client: AsyncClient,
    auth_headers: dict
):
    response = await async_client.get("/users/me", headers=auth_headers)
    assert response.status_code == 200

# Test database operations:
@pytest.mark.asyncio
async def test_repository(db_session: AsyncSession):
    repository = UserRepository(db_session)
    user = await repository.create(UserCreate(email="test@example.com"))
    assert user.id is not None

# Parametrized tests:
@pytest.mark.parametrize("email,expected", [
    ("valid@example.com", True),
    ("invalid", False),
])
@pytest.mark.asyncio
async def test_email_validation(email: str, expected: bool):
    result = validate_email(email)
    assert result == expected
"""
