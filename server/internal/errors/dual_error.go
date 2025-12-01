package errors

const (
	SearchFailedErr    = "QDRANT_FAILED"
	LLMTimeoutErr      = "LLM_TIMEOUT"
	LLMUnavailableErr  = "LLM_UNAVAILABLE"
	LLMUnhealthErr     = "LLM_UNHEALTH"
	PSQLFailedErr      = "POSTGRES_FAILED"
	NothingFoundErr    = "NOTHING_FOUND"
	CoreUnavailableErr = "CORE_UNAVAILABLE"
)

const (
	searchFailedMsg    = "Не удалось выполнить поиск. Попробуйте позже."
	lLMTimeoutMsg      = "Сервис сейчас очень нагружен. Попробуйте позже."
	lLMUnavailableMsg  = "Не удалось выполнить запрос. Попробуйте позже."
	lLMUnhealthMsg     = "Ой, кажется что-то сломалось. Скоро все починим!"
	psqlFailedMsg      = "Не удалось отправить отзыв. Попробуйте позже."
	nothingFoundMsg    = "Не удалось найти информацию по вашему запросу. Попробуйте переформулировать вопрос."
	CoreUnavailableMsg = "Не удалось выполнить запрос. Попробуйте позже"
)

var messages = map[string]string{
	SearchFailedErr:    searchFailedMsg,
	LLMTimeoutErr:      lLMTimeoutMsg,
	LLMUnavailableErr:  lLMUnavailableMsg,
	LLMUnhealthErr:     lLMUnhealthMsg,
	PSQLFailedErr:      psqlFailedMsg,
	NothingFoundErr:    nothingFoundMsg,
	CoreUnavailableErr: CoreUnavailableMsg,
}

type DualError interface {
	error
	Public() string
	Internal() error
}

type dualError struct {
	internal error
	public   *PublicError
}

type PublicError struct {
	Code    string
	Message string
}

func NewDualError(err error, code string) DualError {
	return &dualError{
		internal: err,
		public: &PublicError{
			Code:    code,
			Message: messages[code],
		},
	}
}

func (e *dualError) Error() string {
	return e.internal.Error()
}

func (e *dualError) Public() string {
	return e.public.Message
}

func (e *dualError) Internal() error {
	return e.internal
}
