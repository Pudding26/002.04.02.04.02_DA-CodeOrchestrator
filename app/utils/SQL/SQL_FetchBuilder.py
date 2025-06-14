from sqlalchemy import not_, select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import InstrumentedAttribute
from typing import Type, Optional


class SQL_FetchBuilder:
    """
    Builds SQLAlchemy Select objects for fetching model data.
    Supports method='all' or 'filter' with contains/notcontains filters.
    """

    def __init__(self, orm_class: Type, filter_dict: dict = None):
        self.orm_class = orm_class
        self.filter_dict = filter_dict or {}

    def build_select(self, method: str, columns: Optional[list[str]] = None):
        if method == "all":
            return self._build_all_select(columns)
        elif method == "filter":
            return self._build_filter_select(columns)
        else:
            raise ValueError(f"Unsupported fetch method: {method}")

    def _build_all_select(self, columns: Optional[list[str]] = None):
        selected = self._resolve_columns(columns)
        return select(*selected)

    def _build_filter_select(self, columns: Optional[list[str]] = None):
        if not self.filter_dict:
            raise ValueError("Filter method requires a non-empty filter_dict")

        selected = self._resolve_columns(columns)
        stmt = select(*selected)

        # Apply contains filters
        for column, values in self.filter_dict.get("contains", {}).items():
            if values:
                col = getattr(self.orm_class, column, None)
                if not isinstance(col, InstrumentedAttribute):
                    raise ValueError(f"{column} is not a valid ORM column")
                stmt = stmt.where(col.in_(values))

        # Apply notcontains filters
        for column, values in self.filter_dict.get("notcontains", {}).items():
            if values:
                col = getattr(self.orm_class, column, None)
                if not isinstance(col, InstrumentedAttribute):
                    raise ValueError(f"{column} is not a valid ORM column")
                stmt = stmt.where(not_(col.in_(values)))

        return stmt

    def _resolve_columns(self, columns: Optional[list[str]]):
        """
        Resolves a list of column names into ORM column objects.
        If columns is None, selects all columns from the table.
        """
        if columns:
            try:
                return [getattr(self.orm_class, col) for col in columns]
            except AttributeError as e:
                raise ValueError(f"Invalid column name in projection: {e}")
        else:
            return list(self.orm_class.__table__.columns)
