"""
Task Manager
비동기 작업 상태 관리
"""

from typing import Dict, Any, Optional
from datetime import datetime
import threading


class TaskManager:
    """작업 상태를 관리하는 매니저"""

    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_task(self, task_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        새 작업 생성

        Args:
            task_id: 작업 ID
            data: 작업 데이터

        Returns:
            생성된 작업 정보
        """
        with self._lock:
            task = {
                "task_id": task_id,
                "status": "processing",
                "progress": 0,
                "message": "작업 시작",
                "data": data,
                "result": None,
                "error": None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            self._tasks[task_id] = task
            return task

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        작업 정보 조회

        Args:
            task_id: 작업 ID

        Returns:
            작업 정보 또는 None
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                # 민감한 데이터 제외하고 반환
                return {
                    "task_id": task["task_id"],
                    "status": task["status"],
                    "progress": task["progress"],
                    "message": task["message"],
                    "result": task["result"],
                    "error": task["error"],
                    "created_at": task["created_at"],
                    "updated_at": task["updated_at"]
                }
            return None

    def update_progress(
        self,
        task_id: str,
        progress: float,
        message: str = ""
    ) -> bool:
        """
        작업 진행률 업데이트

        Args:
            task_id: 작업 ID
            progress: 진행률 (0-100)
            message: 상태 메시지

        Returns:
            성공 여부
        """
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["progress"] = min(max(progress, 0), 100)
                if message:
                    self._tasks[task_id]["message"] = message
                self._tasks[task_id]["updated_at"] = datetime.now().isoformat()
                return True
            return False

    def complete_task(
        self,
        task_id: str,
        result: Dict[str, Any]
    ) -> bool:
        """
        작업 완료 처리

        Args:
            task_id: 작업 ID
            result: 결과 데이터

        Returns:
            성공 여부
        """
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = "completed"
                self._tasks[task_id]["progress"] = 100
                self._tasks[task_id]["message"] = "완료"
                self._tasks[task_id]["result"] = result
                self._tasks[task_id]["updated_at"] = datetime.now().isoformat()
                return True
            return False

    def fail_task(
        self,
        task_id: str,
        error: str
    ) -> bool:
        """
        작업 실패 처리

        Args:
            task_id: 작업 ID
            error: 에러 메시지

        Returns:
            성공 여부
        """
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = "failed"
                self._tasks[task_id]["message"] = "실패"
                self._tasks[task_id]["error"] = error
                self._tasks[task_id]["updated_at"] = datetime.now().isoformat()
                return True
            return False

    def cancel_task(self, task_id: str) -> bool:
        """
        작업 취소

        Args:
            task_id: 작업 ID

        Returns:
            성공 여부
        """
        with self._lock:
            if task_id in self._tasks:
                if self._tasks[task_id]["status"] == "processing":
                    self._tasks[task_id]["status"] = "cancelled"
                    self._tasks[task_id]["message"] = "취소됨"
                    self._tasks[task_id]["updated_at"] = datetime.now().isoformat()
                    return True
            return False

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """
        오래된 작업 정리

        Args:
            max_age_hours: 최대 보관 시간 (시간)
        """
        from datetime import timedelta

        with self._lock:
            now = datetime.now()
            to_delete = []

            for task_id, task in self._tasks.items():
                created = datetime.fromisoformat(task["created_at"])
                if now - created > timedelta(hours=max_age_hours):
                    to_delete.append(task_id)

            for task_id in to_delete:
                del self._tasks[task_id]

    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """모든 작업 목록 반환 (관리자용)"""
        with self._lock:
            return {
                task_id: {
                    "task_id": task["task_id"],
                    "status": task["status"],
                    "progress": task["progress"],
                    "created_at": task["created_at"]
                }
                for task_id, task in self._tasks.items()
            }
