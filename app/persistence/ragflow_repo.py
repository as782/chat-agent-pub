"""RAGFlow 映射仓储模块。

负责保存本系统与 RAGFlow 数据集的映射关系，不承担检索策略和召回决策。
当前阶段不负责事务提交，由上层调用方统一控制事务边界。
"""

from datetime import datetime
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.base import get_utc_now
from app.persistence.models import RagflowDatasetEntity


class RagflowRepository:
    """RAGFlow 数据集映射仓储。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self._db_session = db_session

    async def upsert_dataset(
        self,
        *,
        dataset_id: str,
        dataset_name: str,
        is_enabled: bool = True,
        dataset_metadata: dict[str, Any] | None = None,
        updated_at: datetime | None = None,
    ) -> RagflowDatasetEntity:
        """创建或更新 RAGFlow 数据集映射。"""

        dataset_entity = await self.get_by_dataset_id(dataset_id)
        if dataset_entity is None:
            current_time = updated_at or get_utc_now()
            dataset_entity = RagflowDatasetEntity(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                is_enabled=is_enabled,
                dataset_metadata=dataset_metadata or {},
                created_at=current_time,
                updated_at=current_time,
            )
            self._db_session.add(dataset_entity)
        else:
            dataset_entity.dataset_name = dataset_name
            dataset_entity.is_enabled = is_enabled
            dataset_entity.dataset_metadata = dataset_metadata or {}
            dataset_entity.updated_at = updated_at or get_utc_now()

        await self._db_session.flush()
        await self._db_session.refresh(dataset_entity)
        return dataset_entity

    async def get_by_dataset_id(self, dataset_id: str) -> RagflowDatasetEntity | None:
        """按数据集标识查询映射记录。"""

        return await self._db_session.get(RagflowDatasetEntity, dataset_id)

    async def list_datasets(
        self,
        *,
        is_enabled: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RagflowDatasetEntity]:
        """分页查询数据集映射列表。"""

        statement: Select[tuple[RagflowDatasetEntity]] = (
            select(RagflowDatasetEntity)
            .order_by(RagflowDatasetEntity.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if is_enabled is not None:
            statement = statement.where(RagflowDatasetEntity.is_enabled == is_enabled)

        result = await self._db_session.execute(statement)
        return list(result.scalars().all())
