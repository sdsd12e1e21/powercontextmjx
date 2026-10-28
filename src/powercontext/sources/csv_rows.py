# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CSV table capture with stable, primary-key-based row evidence."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class CSVDocumentInput(BaseModel):
    """A complete offline CSV document and the key column identifying its rows."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    primary_key: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("document_id must be non-empty and have no surrounding whitespace")  # noqa: TRY003
        return value

    @field_validator("primary_key")
    @classmethod
    def validate_primary_key(cls, value: str) -> str:
        if not value:
            raise ValueError("primary_key must be non-empty")  # noqa: TRY003
        return value


class CSVRowInput(BaseModel):
    """One validated CSV record, preserving the original column order and strings."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_row_shape(self) -> CSVRowInput:
        if not self.document_id or self.document_id != self.document_id.strip():
            raise ValueError("document_id must be non-empty and have no surrounding whitespace")  # noqa: TRY003
        if not self.primary_key:
            raise ValueError("primary_key must be non-empty")  # noqa: TRY003
        if len(self.columns) != len(self.values):
            raise ValueError("columns and values must have the same length")  # noqa: TRY003
        if self.primary_key not in self.columns:
            raise ValueError("primary_key must exactly match a column name")  # noqa: TRY003
        if not self.primary_key_value.strip():
            raise ValueError("primary key value must not be blank")  # noqa: TRY003
        return self


class CSVRowSource(Source):
    """Captured row source with a stable identity derived from document and key."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if not value or len(value) > 256:
            raise ValueError("source_id must be non-empty and at most 256 characters")  # noqa: TRY003
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> CSVRowSource:
        expected = csv_row_source_id(self.document_id, self.primary_key_value)
        if self.source_id != expected or self.name != expected:
            raise ValueError("name and source_id must match the CSV row identity")  # noqa: TRY003
        if len(self.columns) != len(self.values):
            raise ValueError("columns and values must have the same length")  # noqa: TRY003
        if self.primary_key not in self.columns:
            raise ValueError("primary_key must exactly match a column name")  # noqa: TRY003
        return self


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Return the canonical, delimiter-safe identity for a CSV row."""

    if not isinstance(document_id, str) or not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be non-empty and have no surrounding whitespace")  # noqa: TRY003
    if not isinstance(primary_key_value, str) or not primary_key_value.strip():
        raise ValueError("primary_key_value must not be empty or blank")  # noqa: TRY003
    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > 256:
        raise ValueError("CSV row source identity must not exceed 256 characters")  # noqa: TRY003
    return source_id


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and validate a whole CSV document atomically into row inputs."""

    if not isinstance(value, CSVDocumentInput):
        raise ValueError("value must be a CSVDocumentInput")  # noqa: TRY003
    # Validate the identity components before parsing any records.
    if not value.document_id or value.document_id != value.document_id.strip():
        raise ValueError("document_id must be non-empty and have no surrounding whitespace")  # noqa: TRY003
    if not value.primary_key:
        raise ValueError("primary_key must be non-empty")  # noqa: TRY003

    try:
        text = value.content.decode("utf-8-sig", errors="strict")
        reader = csv.reader(
            io.StringIO(text, newline=""),
            delimiter=",",
            quotechar='"',
            doublequote=True,
            strict=True,
        )
        records = iter(reader)
        try:
            header = next(records)
        except StopIteration:
            raise ValueError("CSV document must contain a header") from None

        if not header:
            raise ValueError("CSV header must not be empty")  # noqa: TRY003
        if any(not column or not column.strip() for column in header):
            raise ValueError("CSV header fields must be non-empty and non-blank")  # noqa: TRY003
        if len(set(header)) != len(header):
            raise ValueError("CSV header fields must be unique")  # noqa: TRY003
        try:
            key_index = header.index(value.primary_key)
        except ValueError:
            raise ValueError("primary_key must exactly match a CSV header field") from None

        parsed: list[CSVRowInput] = []
        seen_keys: set[str] = set()
        columns = tuple(header)
        for record in records:
            # csv.reader represents an entirely blank physical line as [].
            if record == []:
                continue
            if len(record) != len(columns):
                raise ValueError("CSV record width must match the header width")  # noqa: TRY003
            key_value = record[key_index]
            if not key_value.strip():
                raise ValueError("CSV primary key values must not be blank")  # noqa: TRY003
            if key_value in seen_keys:
                raise ValueError("CSV primary key values must be unique within a document")  # noqa: TRY003
            seen_keys.add(key_value)
            # Ensure even an overlong identity fails before any partial result escapes.
            csv_row_source_id(value.document_id, key_value)
            parsed.append(
                CSVRowInput(
                    document_id=value.document_id,
                    primary_key=value.primary_key,
                    primary_key_value=key_value,
                    columns=columns,
                    values=tuple(record),
                )
            )
        return tuple(parsed)
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("invalid UTF-8 or malformed CSV document") from error


class CSVRowAdapter:
    """Resolve and read row inputs as captured Sources."""

    input_class = CSVRowInput
    name = "csv-row"
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
            materialization=SourceMaterialization.CAPTURED,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Project a row into canonical JSON evidence for Memory indexing."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type="csv-row",
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = AdapterSourceDefinition(
    CSVRowAdapter(),
    version="1",
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
