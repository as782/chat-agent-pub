"""会话服务模块。

负责会话创建、查询和列表等业务编排，不承担底层数据库访问实现。
当前阶段不负责会话标题生成策略和复杂状态流转。
"""

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ResourceNotFoundException
from app.persistence.models import SessionEntity
from app.persistence.session_repo import SessionRepository
from app.schemas.session import SessionCreateRequest, SessionListResponse, SessionResponse


class SessionService:
    """会话服务。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self._db_session = db_session
        self._session_repository = SessionRepository(db_session)

    async def create_session(self, request: SessionCreateRequest) -> SessionResponse:
        """创建新会话。"""

        try:
            session_entity = await self._session_repository.create(
                session_id=self._generate_identifier(),
                title=request.title,
                user_id=request.user_id,
            )
            await self._db_session.commit()
        except Exception:
            await self._db_session.rollback()
            raise

        return self._to_session_response(session_entity)

    async def get_session(self, session_id: str) -> SessionResponse:
        """查询单个会话。"""

        session_entity = await self._session_repository.get_by_id(session_id)
        if session_entity is None:
            raise ResourceNotFoundException(
                "会话不存在",
                details={"session_id": session_id},
            )
        return self._to_session_response(session_entity)

    async def list_sessions(
        self,
        *,
        user_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> SessionListResponse:
        """分页查询会话列表。"""

        session_entities = await self._session_repository.list(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )
        total = await self._session_repository.count(user_id=user_id)
        response_items = [
            self._to_session_response(session_entity) for session_entity in session_entities
        ]
        return SessionListResponse(
            items=response_items,
            total=total,
        )

    @staticmethod
    def _to_session_response(session_entity: SessionEntity) -> SessionResponse:
        """将 ORM 对象转换为对外响应模型。"""

        return SessionResponse(
            session_id=session_entity.session_id,
            title=session_entity.title,
            user_id=session_entity.user_id,
            status=session_entity.status,
            created_at=session_entity.created_at,
            updated_at=session_entity.updated_at,
        )

    @staticmethod
    def _generate_identifier() -> str:
        """生成统一长度的业务标识。"""

        return uuid4().hex
