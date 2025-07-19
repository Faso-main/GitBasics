from sqlalchemy.orm import Session

from BD.Models import QuestAnswer


class SqlFunctions:

    @staticmethod
    def add_data(question: str, answer: str, session: Session) -> None:
        """
        Добавляет данные в любом случае

        :param question: Вопрос, для которого нужно добавить записи.
        :param answer: Ответ к вопросу
        :param session: Объект сессии SQLAlchemy.
        """
        new_data = QuestAnswer(question=question, answer=answer)
        session.add(new_data)
        session.commit()

    # Функция для получения всех данных
    @staticmethod
    def get_all_data(session: Session):
        return session.query(QuestAnswer).all()

    @staticmethod
    def add_if_not_exists(question: str, answer: str, session: Session) -> None:
        """
        Добавляет данные если в бд нет данного вопроса

        :param question: Вопрос, для которого нужно добавить записи.
        :param answer: Ответ к вопросу
        :param session: Объект сессии SQLAlchemy.
        """
        # Проверка существования записи в базе данных
        existing_entry = session.query(QuestAnswer).filter(
            QuestAnswer.question == question
        ).first()

        if existing_entry is None:  # Если запись не найдена
            new_entry = QuestAnswer(question=question, answer=answer)
            session.add(new_entry)
            session.commit()
            print(f"Добавлено: вопрос '{question}' с ответом '{answer}'.")
        else:
            print(f"Запись с вопросом '{question}' уже существует.")

    @staticmethod
    def delete_answers_by_question(question, session):
        """
        Удаляет всей записи из базы данных, соответствующие заданному вопросу.

        :param question: Вопрос, для которого нужно удалить записи.
        :param session: Объект сессии SQLAlchemy.
        """
        try:
            # Выполняем запрос на удаление
            deleted_count = session.query(QuestAnswer).filter(QuestAnswer.question == question).delete()

            session.commit()  # Фиксация изменений
            print(f"Удалено записей: {deleted_count}")
        except Exception as e:
            session.rollback()  # Откат при ошибке
            print(f"Произошла ошибка: {e}")