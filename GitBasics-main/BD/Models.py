import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# Создаем базовый класс для модели
Base = declarative_base()


# Определяем модель таблицы 'QuestAnswer'
class QuestAnswer(Base):
    __tablename__ = 'QuestAnswer'

    id = sa.Column(sa.Integer, primary_key=True)
    question = sa.Column(sa.String, nullable=False)
    answer = sa.Column(sa.String, nullable=False)

    def __repr__(self):
        return f"id={self.id}, question='{self.question}', answer='{self.answer}'"

"""
class Group(Base):
    __tablename__ = 'groups'

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String, nullable=False)

    # Отношение: одна группа может иметь много расписаний
    schedules = relationship('Schedule', back_populates='group')


class Subject(Base):
    __tablename__ = 'subjects'

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String, nullable=False)

    # Отношение: один предмет может быть в расписании многих групп
    schedules = relationship('Schedule', back_populates='subject')


class Teacher(Base):
    __tablename__ = 'teachers'

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String, nullable=False)

    # Отношение: один преподаватель может вести много предметов
    subjects = relationship('Subject', back_populates='teachers')


# Расписание для групп
class Schedule(Base):
    __tablename__ = 'schedules'

    id = sa.Column(sa.Integer, primary_key=True)
    group_id = sa.Column(sa.Integer, sa.ForeignKey('groups.id'), nullable=False)
    subject_id = sa.Column(sa.Integer,sa.ForeignKey('subjects.id'), nullable=False)
    teacher_id = sa.Column(sa.Integer, sa.ForeignKey('teachers.id'), nullable=False)
    start_time = sa.Column(sa.Time, nullable=False)
    end_time = sa.Column(sa.Time, nullable=False)
    day_of_week = sa.Column(sa.String, nullable=False)  # например, 'Monday', 'Tuesday', 'Wednesday' и т. д.

    # Определяем отношения
    group = relationship('Group', back_populates='schedules')
    subject = relationship('Subject', back_populates='schedules')
    teacher = relationship('Teacher', back_populates='subjects')
"""
