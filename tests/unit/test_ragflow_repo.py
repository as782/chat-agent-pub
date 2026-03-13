"""RAGFlow 仓储单元测试。"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.ragflow_repo import RagflowRepository


@pytest.mark.asyncio
async def test_ragflow_repository_upsert_creates_and_updates_dataset(
    db_session: AsyncSession,
) -> None:
    """验证数据集映射支持创建与更新。"""

    repository = RagflowRepository(db_session)

    await repository.upsert_dataset(
        dataset_id="dataset-001",
        dataset_name="知识库 A",
        is_enabled=True,
        dataset_metadata={"space": "default"},
    )
    dataset = await repository.upsert_dataset(
        dataset_id="dataset-001",
        dataset_name="知识库 A-已更新",
        is_enabled=False,
        dataset_metadata={"space": "updated"},
    )

    assert dataset.dataset_name == "知识库 A-已更新"
    assert dataset.is_enabled is False
    assert dataset.dataset_metadata == {"space": "updated"}


@pytest.mark.asyncio
async def test_ragflow_repository_list_datasets_supports_enabled_filter(
    db_session: AsyncSession,
) -> None:
    """验证数据集列表支持启用状态过滤。"""

    repository = RagflowRepository(db_session)

    await repository.upsert_dataset(
        dataset_id="dataset-001",
        dataset_name="启用数据集",
        is_enabled=True,
    )
    await repository.upsert_dataset(
        dataset_id="dataset-002",
        dataset_name="停用数据集",
        is_enabled=False,
    )

    enabled_datasets = await repository.list_datasets(is_enabled=True)

    assert [dataset.dataset_id for dataset in enabled_datasets] == ["dataset-001"]
